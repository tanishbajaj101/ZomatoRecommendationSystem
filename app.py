import os
import ast
import json
import math
import sqlite3
import uuid
from typing import List, Dict, Tuple, Set

import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
from streamlit_geolocation import st_geolocation  # <-- use this component

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# ====== CONFIG ======
DB_PATH = "restaurants.db"   # your SQLite file
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
RESTAURANT_COLLECTION = "restaurants"
USER_PROFILE_COLLECTION = "user_profiles"

# SentenceTransformer model directory (uploaded to Streamlit Cloud storage or mounted)
MODEL_LOCAL_DIR = os.getenv("MODEL_LOCAL_DIR", "./models/all-MiniLM-L6-v2")

DEFAULT_WEIGHTS = {"cuisine": 0.35, "ambience": 0.25, "rating": 0.20, "price": 0.20}
DEFAULT_PRICE_PREF = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)  # [0..3], sum=1
DEFAULT_USER_TEXT_CUISINE = "popular dishes, Indian, Chinese, Continental, comfort food"
DEFAULT_USER_TEXT_AMBIENCE = "cozy, clean, family friendly, natural ambience"

# Default fallback location (Delhi CP area) if geolocation is unavailable
DEFAULT_LAT, DEFAULT_LON = 28.6315, 77.2167
RADIUS_KM = 10.0
TOP_K = 8


def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    a = np.array([
        w.get("cuisine", 0.0),
        w.get("ambience", 0.0),
        w.get("rating", 0.0),
        w.get("price", 0.0),
    ], dtype=float)
    s = float(a.sum())
    if s <= 0:
        return DEFAULT_WEIGHTS.copy()
    a = a / s
    return {"cuisine": float(a[0]), "ambience": float(a[1]), "rating": float(a[2]), "price": float(a[3])}


def user_point_id(user_id: str) -> str:
    # deterministic UUIDv5 from your logical user id
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"user/{user_id}"))


# ====== LAZY IMPORT FOR EMBEDDINGS ======
@st.cache_resource(show_spinner=False)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    # Streamlit Cloud friendly: operate offline if model is already present
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return SentenceTransformer(MODEL_LOCAL_DIR)


# ====== QDRANT CLIENT + COLLECTIONS ======
@st.cache_resource(show_spinner=False)
def get_qdrant():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def ensure_collections(client: QdrantClient, dim: int = 384):
    # restaurants collection (two named dense vectors)
    if not client.collection_exists(collection_name=RESTAURANT_COLLECTION):
        client.create_collection(
            collection_name=RESTAURANT_COLLECTION,
            vectors_config={  # version-agnostic: dict or VectorsConfig is OK
                "cuisine": qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
                "ambience": qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            },
            optimizers_config=qmodels.OptimizersConfigDiff(indexing_threshold=1000),
        )

    # user profiles collection
    if not client.collection_exists(collection_name=USER_PROFILE_COLLECTION):
        client.create_collection(
            collection_name=USER_PROFILE_COLLECTION,
            vectors_config={
                "cuisine": qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
                "ambience": qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            },
        )

    # GEO payload index on "location" (idempotent)
    try:
        client.create_payload_index(
            collection_name=RESTAURANT_COLLECTION,
            field_name="location",
            field_schema=qmodels.PayloadSchemaType.GEO,
        )
    except Exception:
        pass  # already exists


# ====== SQLITE ======
@st.cache_resource(show_spinner=False)
def get_conn(db_path: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_rows_by_ids(conn, ids: List[int]) -> List[sqlite3.Row]:
    if not ids:
        return []
    q = f"SELECT * FROM restaurants WHERE idx IN ({','.join('?' for _ in ids)})"
    return conn.execute(q, ids).fetchall()


# ====== VECTORIZATION / UPSERT ======
def combine_cuisine_food(cuisine: str, food_items: str) -> str:
    c = cuisine or ""
    f = food_items or ""
    return f"Cuisines: {c}. Signature items: {f}."


def vectorize_restaurant(embedder, cuisine: str, food_items: str, ambience: str) -> Tuple[np.ndarray, np.ndarray]:
    text_c = combine_cuisine_food(cuisine, food_items)
    text_a = ambience or ""
    v_c = embedder.encode(text_c, normalize_embeddings=True)
    v_a = embedder.encode(text_a, normalize_embeddings=True)
    return v_c.astype(np.float32), v_a.astype(np.float32)


def ensure_qdrant_synced_from_sqlite(conn, client, embedder, limit: int = None):
    rows = conn.execute(
        "SELECT idx, Cuisine, Food_Items, Ambience, Latitude, Longitude, Bayesian_Rating, Price_Category FROM restaurants"
    ).fetchall()
    if limit is not None:
        rows = rows[:limit]
    points = []
    for r in rows:
        idx = int(r["idx"])
        lat = float(r["Latitude"])
        lon = float(r["Longitude"])
        v_c, v_a = vectorize_restaurant(embedder, r["Cuisine"], r["Food_Items"], r["Ambience"])
        payload = {
            "idx": idx,
            "lat": lat,
            "lon": lon,
            "location": {"lat": lat, "lon": lon},  # GEO field for Qdrant geo index
            "bayes": float(r["Bayesian_Rating"] or 0.0),
            "price": int(r["Price_Category"] or 0),
        }
        points.append(
            qmodels.PointStruct(
                id=idx,
                vector={"cuisine": v_c.tolist(), "ambience": v_a.tolist()},
                payload=payload,
            )
        )
        if len(points) >= 256:
            client.upsert(RESTAURANT_COLLECTION, points=points)
            points = []
    if points:
        client.upsert(RESTAURANT_COLLECTION, points=points)


# ====== USER PROFILE VECTORS & INTERACTIONS ======
def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def _empty_interactions() -> Dict[str, List[int]]:
    return {"liked": [], "disliked": []}


def get_or_bootstrap_user_profile(embedder, client: QdrantClient, user_id: str):
    pid = user_point_id(user_id)
    try:
        pts = client.retrieve(USER_PROFILE_COLLECTION, ids=[pid], with_vectors=True, with_payload=True)
        if pts:
            v_c = np.array(pts[0].vector["cuisine"], dtype=np.float32)
            v_a = np.array(pts[0].vector["ambience"], dtype=np.float32)
            pl = pts[0].payload or {}
            price = np.array(pl.get("price_pref", DEFAULT_PRICE_PREF.tolist()), dtype=float)
            has_fb = bool(pl.get("has_feedback", False))
            w = pl.get("score_weights", DEFAULT_WEIGHTS.copy())
            w = normalize_weights(w)
            inter = pl.get("interactions", _empty_interactions())
            inter = {
                "liked": [int(x) for x in inter.get("liked", [])],
                "disliked": [int(x) for x in inter.get("disliked", [])],
            }
            return {"cuisine": v_c, "ambience": v_a}, price, has_fb, w, inter
    except Exception:
        pass

    # cold start
    v_c = embedder.encode(DEFAULT_USER_TEXT_CUISINE, normalize_embeddings=True).astype(np.float32)
    v_a = embedder.encode(DEFAULT_USER_TEXT_AMBIENCE, normalize_embeddings=True).astype(np.float32)
    price_pref = DEFAULT_PRICE_PREF.copy()
    weights = DEFAULT_WEIGHTS.copy()
    interactions = _empty_interactions()
    client.upsert(
        USER_PROFILE_COLLECTION,
        points=[qmodels.PointStruct(
            id=pid,
            vector={"cuisine": v_c.tolist(), "ambience": v_a.tolist()},
            payload={
                "user_id": user_id,
                "price_pref": price_pref.tolist(),
                "has_feedback": False,
                "score_weights": weights,
                "interactions": interactions,
            },
        )],
    )
    return {"cuisine": v_c, "ambience": v_a}, price_pref, False, weights, interactions


def update_user_profile(client: QdrantClient, user_id: str,
                        user_vecs: Dict[str, np.ndarray],
                        price_pref: np.ndarray,
                        has_feedback: bool,
                        weights: Dict[str, float],
                        interactions: Dict[str, List[int]] = None):
    pid = user_point_id(user_id)
    weights = normalize_weights(weights or DEFAULT_WEIGHTS)
    if interactions is None:
        interactions = _empty_interactions()
    client.upsert(
        USER_PROFILE_COLLECTION,
        points=[qmodels.PointStruct(
            id=pid,
            vector={"cuisine": user_vecs["cuisine"].tolist(), "ambience": user_vecs["ambience"].tolist()},
            payload={
                "user_id": user_id,
                "price_pref": price_pref.tolist(),
                "has_feedback": has_feedback,
                "score_weights": weights,
                "interactions": interactions,
            },
        )],
    )


def record_interaction(client: QdrantClient, user_id: str, rest_id: int, liked: bool):
    inter = st.session_state.get("interactions", _empty_interactions())
    if liked:
        if rest_id not in inter["liked"]:
            inter["liked"].append(rest_id)
        if rest_id in inter["disliked"]:
            inter["disliked"].remove(rest_id)
    else:
        if rest_id not in inter["disliked"]:
            inter["disliked"].append(rest_id)
        if rest_id in inter["liked"]:
            inter["liked"].remove(rest_id)
    st.session_state["interactions"] = inter
    hidden = st.session_state.get("hidden_ids", set())
    hidden.add(int(rest_id))
    st.session_state["hidden_ids"] = hidden
    update_user_profile(
        client,
        st.session_state["user_id"],
        st.session_state["user_vecs"],
        st.session_state["price_pref"],
        st.session_state["has_feedback"],
        st.session_state.get("weights", DEFAULT_WEIGHTS.copy()),
        interactions=inter,
    )


def remove_interaction(client: QdrantClient, user_id: str, rest_id: int):
    inter = st.session_state.get("interactions", _empty_interactions())
    rest_id = int(rest_id)
    changed = False
    if rest_id in inter["liked"]:
        inter["liked"].remove(rest_id)
        changed = True
    if rest_id in inter["disliked"]:
        inter["disliked"].remove(rest_id)
        changed = True
    if changed:
        st.session_state["interactions"] = inter
        hidden = st.session_state.get("hidden_ids", set())
        if rest_id in hidden:
            hidden.remove(rest_id)
        st.session_state["hidden_ids"] = hidden
        update_user_profile(
            client,
            st.session_state["user_id"],
            st.session_state["user_vecs"],
            st.session_state["price_pref"],
            st.session_state["has_feedback"],
            st.session_state.get("weights", DEFAULT_WEIGHTS.copy()),
            interactions=inter,
        )


def clear_all_interactions(client: QdrantClient, user_id: str):
    st.session_state["interactions"] = _empty_interactions()
    st.session_state["hidden_ids"] = set()
    update_user_profile(
        client,
        st.session_state["user_id"],
        st.session_state["user_vecs"],
        st.session_state["price_pref"],
        st.session_state["has_feedback"],
        st.session_state.get("weights", DEFAULT_WEIGHTS.copy()),
        interactions=st.session_state["interactions"],
    )


# ====== RECOMMENDATION (with Qdrant GEO filter) ======
def recommend(client: QdrantClient,
              embedder,
              conn,
              user_loc: Tuple[float, float],
              user_vecs: Dict[str, np.ndarray],
              price_pref: np.ndarray,
              weights: Dict[str, float],
              top_k: int = TOP_K,
              excluded_ids: Set[int] = None):
    geo_filter = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="location",
                geo_radius=qmodels.GeoRadius(
                    center=qmodels.GeoPoint(lat=float(user_loc[0]), lon=float(user_loc[1])),
                    radius=float(RADIUS_KM * 1000.0),
                ),
            )
        ]
    )

    limit = max(200, top_k * 10)

    srch = [
        qmodels.SearchRequest(
            vector=qmodels.NamedVector(name="cuisine", vector=user_vecs["cuisine"].tolist()),
            limit=limit,
            filter=geo_filter,
        ),
        qmodels.SearchRequest(
            vector=qmodels.NamedVector(name="ambience", vector=user_vecs["ambience"].tolist()),
            limit=limit,
            filter=geo_filter,
        ),
    ]

    res = client.search_batch(RESTAURANT_COLLECTION, srch)

    sc_c = {pt.id: float(pt.score) for pt in res[0]}
    sc_a = {pt.id: float(pt.score) for pt in res[1]}
    ids = list(set(list(sc_c.keys()) + list(sc_a.keys())))
    if not ids:
        return []

    rows = fetch_rows_by_ids(conn, ids)

    ex = excluded_ids or set()
    ranked = []
    for r in rows:
        idx = int(r["idx"])
        if idx in ex:
            continue
        cuisine_s = sc_c.get(idx, 0.0)
        amb_s = sc_a.get(idx, 0.0)
        bayes = float(r["Bayesian_Rating"] or 0.0)
        bayes_norm = max(0.0, min(1.0, bayes / 5.0))
        price_cat = int(r["Price_Category"] or 0)
        price_ctx = float(price_pref[price_cat])
        final = (
            weights["cuisine"] * cuisine_s +
            weights["ambience"] * amb_s +
            weights["rating"] * bayes_norm +
            weights["price"] * price_ctx
        )
        ranked.append((final, r))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in ranked[:top_k]]


# ====== DISPLAY-ONLY GEO UTILS ======
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in kilometers between two WGS84 points."""
    R = 6371.0088
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlmb = math.radians(float(lon2) - float(lon1))
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2)
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


# ====== IMAGE CAROUSEL ======
def parse_images_field(images_value) -> List[str]:
    if images_value is None:
        return []
    if isinstance(images_value, list):
        return images_value
    s = str(images_value)
    try:
        val = json.loads(s)
        if isinstance(val, list):
            return [str(u) for u in val]
    except Exception:
        pass
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [str(u) for u in val]
    except Exception:
        pass
    return [s]  # fallback single


def image_carousel(urls: List[str], key: str):
    if key not in st.session_state:
        st.session_state[key] = 0
    if not urls:
        st.markdown("_(No images)_")
        return
    i = st.session_state[key] % len(urls)
    st.image(urls[i], use_container_width=True)
    c1, c2, c3 = st.columns([1, 1, 6])
    with c1:
        if st.button("◀️", key=f"prev_{key}"):
            st.session_state[key] = (st.session_state[key] - 1) % len(urls)
            st.rerun()
    with c2:
        if st.button("▶️", key=f"next_{key}"):
            st.session_state[key] = (st.session_state[key] + 1) % len(urls)
            st.rerun()
    with c3:
        st.caption(f"{i+1} / {len(urls)}")


# ====== LIKE / DISLIKE ======
def apply_feedback(embedder, client, user_id: str, row: sqlite3.Row, liked: bool,
                   user_vecs: Dict[str, np.ndarray], price_pref: np.ndarray, has_feedback: bool):
    vc, va = vectorize_restaurant(embedder, row["Cuisine"], row["Food_Items"], row["Ambience"])
    pc = int(row["Price_Category"] or 0)

    if not has_feedback:
        # FIRST INTERACTION: hard override of cold-start vectors
        if liked:
            user_vecs["cuisine"] = normalize(vc)
            user_vecs["ambience"] = normalize(va)
            pp = np.full(4, 0.05, dtype=float)
            pp[pc] = 0.85
            price_pref[:] = pp / pp.sum()
        else:
            user_vecs["cuisine"] = normalize(user_vecs["cuisine"] - vc)
            user_vecs["ambience"] = normalize(user_vecs["ambience"] - va)
            pp = np.full(4, 0.25, dtype=float)
            pp[pc] = max(0.05, pp[pc] - 0.15)
            price_pref[:] = pp / pp.sum()
        has_feedback = True
    else:
        alpha = 0.20 if liked else -0.20
        user_vecs["cuisine"] = normalize(user_vecs["cuisine"]*(1 - abs(alpha)) + alpha*vc)
        user_vecs["ambience"] = normalize(user_vecs["ambience"]*(1 - abs(alpha)) + alpha*va)
        delta = 0.10 if liked else -0.08
        price_pref[pc] = max(0.01, price_pref[pc] + delta)
        s = price_pref.sum()
        if s > 0:
            price_pref[:] = price_pref / s

    # Record & persist interaction; hide from future recommendations
    record_interaction(client, user_id, int(row["idx"]), liked)

    # Persist vectors + flags as well (with weights)
    w = st.session_state.get("weights", DEFAULT_WEIGHTS.copy())
    update_user_profile(client, user_id, user_vecs, price_pref, has_feedback, w, st.session_state["interactions"])
    return has_feedback


# ====== UI RENDERING (STRICT FIELDS ONLY + distance badge) ======
def render_restaurant_card(row: sqlite3.Row, user_lat: float = None, user_lon: float = None):
    img_urls = parse_images_field(row["Images"])

    dist_text = None
    try:
        if user_lat is not None and user_lon is not None and row["Latitude"] is not None and row["Longitude"] is not None:
            d_km = haversine_km(user_lat, user_lon, float(row["Latitude"]), float(row["Longitude"]))
            dist_text = f"{d_km:.1f} km"
    except Exception:
        dist_text = None

    with st.container(border=True):
        c_img, c_txt = st.columns([1.2, 2])
        with c_img:
            image_carousel(img_urls, key=f"carousel_{row['idx']}")
        with c_txt:
            # Title + distance badge (you asked to show distance)
            title = str(row["Restaurant_Name"])
            if dist_text:
                st.subheader(title + f"  ·  🧭 {dist_text}")
            else:
                st.subheader(title)

            st.markdown(f"**Cuisine:** {row['Cuisine']}")
            st.markdown(f"**Pricing for 2:** ₹{int(row['Pricing_for_2']) if row['Pricing_for_2'] is not None else '—'}")
            st.markdown(f"**Dining Rating:** {row['Dining_Rating']}")
            st.markdown(f"**Dining Review Count:** {row['Dining_Review_Count']}")
            if row["Website"]:
                st.markdown(f"[Website]({row['Website']})")
            st.markdown(f"**Address:** {row['Address']}")
            st.markdown(f"**Food Items:** {row['Food_Items']}")
            st.markdown(f"**Ambience:** {row['Ambience']}")


def render_interaction_list(conn, ids: List[int], label_empty: str):
    ids = list(dict.fromkeys(int(x) for x in ids))  # de-dup, preserve order
    if not ids:
        st.caption(label_empty)
        return
    rows = fetch_rows_by_ids(conn, ids)
    order = {int(i): k for k, i in enumerate(ids)}
    rows.sort(key=lambda r: order.get(int(r["idx"]), 1e9))
    for r in rows:
        with st.container(border=True):
            c1, c2 = st.columns([2, 1])
            with c1:
                price = f"₹{int(r['Pricing_for_2'])}" if r["Pricing_for_2"] else "—"
                st.markdown(f"**{r['Restaurant_Name']}**  \n_{r['Cuisine']}_ • {price} • ⭐ {r['Dining_Rating']}")
                st.caption(r["Address"])
            with c2:
                if st.button("Unhide", key=f"unhide_{r['idx']}"):
                    remove_interaction(get_qdrant(), st.session_state["user_id"], int(r["idx"]))
                    st.toast("Unhidden from history. It can appear in recommendations again.")
                    st.rerun()


# ====== MAIN APP ======
def main():
    st.set_page_config(page_title="Zomato Recommender", layout="wide")
    st.title("🍽️ Zomato Restaurant Recommender")

    conn = get_conn(DB_PATH)
    embedder = get_embedder()
    client = get_qdrant()
    ensure_collections(client)

    # Sidebar admin (collapsed by default)
    with st.sidebar:
        with st.expander("⚙️ Admin / Setup", expanded=False):
            if st.button("Sync/Refresh Qdrant from SQLite"):
                with st.spinner("Indexing restaurants into Qdrant..."):
                    ensure_qdrant_synced_from_sqlite(conn, client, embedder)
                st.success("Qdrant synced from SQLite.")

            if st.button("Reset profile (cold start)", key="reset_profile_btn"):
                v_c = embedder.encode(DEFAULT_USER_TEXT_CUISINE, normalize_embeddings=True).astype(np.float32)
                v_a = embedder.encode(DEFAULT_USER_TEXT_AMBIENCE, normalize_embeddings=True).astype(np.float32)
                st.session_state["user_vecs"] = {"cuisine": v_c, "ambience": v_a}
                st.session_state["price_pref"] = DEFAULT_PRICE_PREF.copy()
                st.session_state["has_feedback"] = False
                st.session_state["weights"] = DEFAULT_WEIGHTS.copy()
                st.session_state["interactions"] = {"liked": [], "disliked": []}
                st.session_state["hidden_ids"] = set()
                update_user_profile(
                    client,
                    st.session_state.get("user_id", "user-001"),
                    st.session_state["user_vecs"],
                    st.session_state["price_pref"],
                    st.session_state["has_feedback"],
                    st.session_state["weights"],
                    st.session_state["interactions"],
                )
                st.success("Profile reset to cold start.")

    # Session state
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = "user-001"

    if ("user_vecs" not in st.session_state or
        "price_pref" not in st.session_state or
        "has_feedback" not in st.session_state or
        "weights" not in st.session_state or
        "interactions" not in st.session_state):
        uv, pp, hf, w, inter = get_or_bootstrap_user_profile(embedder, client, st.session_state["user_id"])
        st.session_state["user_vecs"] = uv
        st.session_state["price_pref"] = pp
        st.session_state["has_feedback"] = hf
        st.session_state["weights"] = normalize_weights(w)
        st.session_state["interactions"] = inter

    if "hidden_ids" not in st.session_state:
        liked = st.session_state["interactions"].get("liked", [])
        disliked = st.session_state["interactions"].get("disliked", [])
        st.session_state["hidden_ids"] = set([int(x) for x in liked + disliked])

    if "lat" not in st.session_state:
        st.session_state["lat"], st.session_state["lon"] = DEFAULT_LAT, DEFAULT_LON
    if "show_map" not in st.session_state:
        st.session_state["show_map"] = False
    if "pending_loc" not in st.session_state:
        st.session_state["pending_loc"] = None
    if "geo_capture" not in st.session_state:
        st.session_state["geo_capture"] = False  # toggles the st_geolocation component

    # Location UI
    st.markdown("#### 📍 Current Location")
    col_a, col_b, col_c = st.columns([3, 1, 1])
    with col_a:
        st.info(
            f"Latitude: **{st.session_state['lat']:.6f}**, "
            f"Longitude: **{st.session_state['lon']:.6f}** (radius {RADIUS_KM} km)"
        )
    with col_b:
        # When clicked, we render the st_geolocation component below
        if st.button("Use device location"):
            st.session_state["geo_capture"] = True

    with col_c:
        if st.button("Change on map"):
            st.session_state["show_map"] = True

    # Render the device geolocation component on-demand
    if st.session_state["geo_capture"]:
        st.info("Click **Allow** in your browser to share your location.")
        loc = st_geolocation()  # shows a small map + triggers HTML5 geolocation
        # Possible return shapes (depending on version): {'latitude','longitude'} or {'lat','lng'}
        la = lo = None
        if isinstance(loc, dict):
            la = loc.get("latitude", loc.get("lat"))
            lo = loc.get("longitude", loc.get("lng"))
        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            if st.button("Apply device location"):
                if la is not None and lo is not None:
                    st.session_state["lat"], st.session_state["lon"] = float(la), float(lo)
                    st.session_state["show_map"] = False
                    st.session_state["geo_capture"] = False
                    st.success("Device location applied.")
                    st.rerun()
                else:
                    st.warning("Location unavailable yet. Please allow the browser prompt or try again.")
        with col_g2:
            if st.button("Cancel device location"):
                st.session_state["geo_capture"] = False
                st.rerun()

    # ---- Map override flow ----
    if st.session_state["show_map"]:
        m = folium.Map(location=[st.session_state["lat"], st.session_state["lon"]], zoom_start=13)
        folium.Marker([st.session_state["lat"], st.session_state["lon"]], tooltip="Current").add_to(m)
        folium.Circle([st.session_state["lat"], st.session_state["lon"]], radius=RADIUS_KM*1000, fill=False).add_to(m)
        folium.LatLngPopup().add_to(m)

        out = st_folium(m, height=420, key="map", returned_objects=["last_clicked"])

        if out and out.get("last_clicked") is not None:
            la = float(out["last_clicked"]["lat"])
            lo = float(out["last_clicked"]["lng"])
            st.session_state["pending_loc"] = (la, lo)

        if st.session_state["pending_loc"] is not None:
            la, lo = st.session_state["pending_loc"]
            st.info(f"Selected: **{la:.6f}, {lo:.6f}**")
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Use this location", key="apply_loc"):
                    st.session_state["lat"], st.session_state["lon"] = la, lo
                    st.session_state["pending_loc"] = None
                    st.session_state["show_map"] = False
                    st.rerun()
            with c2:
                if st.button("Reset selection", key="reset_loc"):
                    st.session_state["pending_loc"] = None
                    st.rerun()
            with c3:
                if st.button("Cancel", key="cancel_map"):
                    st.session_state["pending_loc"] = None
                    st.session_state["show_map"] = False
                    st.rerun()
        else:
            if st.button("Cancel", key="cancel_map_nopick"):
                st.session_state["show_map"] = False
                st.rerun()

    st.divider()

    # Recommendations + Controls
    left, right = st.columns([2, 1])
    with right:
        # ---- Scoring Weights (normalized & persisted) ----
        st.markdown("#### 🎛️ Scoring Weights")

        if "w_raw" not in st.session_state:
            st.session_state["w_raw"] = (st.session_state.get("weights", DEFAULT_WEIGHTS)).copy()

        wc = st.slider("Cuisine similarity", 0.0, 1.0, float(st.session_state["w_raw"]["cuisine"]), 0.01, key="w_cuisine")
        wa = st.slider("Ambience similarity", 0.0, 1.0, float(st.session_state["w_raw"]["ambience"]), 0.01, key="w_ambience")
        wr = st.slider("Bayesian rating",   0.0, 1.0, float(st.session_state["w_raw"]["rating"]), 0.01, key="w_rating")
        wp = st.slider("Price context",     0.0, 1.0, float(st.session_state["w_raw"]["price"]),  0.01, key="w_price")

        raw_w = {"cuisine": wc, "ambience": wa, "rating": wr, "price": wp}
        current_weights = normalize_weights(raw_w)
        st.session_state["weights"] = current_weights
        st.caption(
            f"Normalized: cuisine {current_weights['cuisine']:.2f} • "
            f"ambience {current_weights['ambience']:.2f} • "
            f"rating {current_weights['rating']:.2f} • "
            f"price {current_weights['price']:.2f}"
        )

        st.markdown("#### 💸 Price Preference (context)")
        p = st.session_state["price_pref"]
        p0 = st.slider("₹ Budget (0)", 0.0, 1.0, float(p[0]), 0.01)
        p1 = st.slider("₹₹ Mid (1)", 0.0, 1.0, float(p[1]), 0.01)
        p2 = st.slider("₹₹₹ Premium (2)", 0.0, 1.0, float(p[2]), 0.01)
        p3 = st.slider("₹₹₹₹ Luxury (3)", 0.0, 1.0, float(p[3]), 0.01)
        s = p0 + p1 + p2 + p3
        if s == 0:
            s = 1.0
        st.session_state["price_pref"] = np.array([p0, p1, p2, p3]) / s

        if st.button("Save preference"):
            hf = st.session_state.get("has_feedback", False)
            update_user_profile(
                client,
                st.session_state["user_id"],
                st.session_state["user_vecs"],
                st.session_state["price_pref"],
                hf,
                current_weights,
                st.session_state["interactions"]
            )
            st.success("Preferences (price + weights) saved.")

        st.markdown("#### ⭐ My Interactions")
        tabs = st.tabs(["❤️ Liked", "🚫 Disliked"])
        with tabs[0]:
            render_interaction_list(conn, st.session_state["interactions"].get("liked", []), "No liked restaurants yet.")
        with tabs[1]:
            render_interaction_list(conn, st.session_state["interactions"].get("disliked", []), "No disliked restaurants yet.")
        c1, _ = st.columns(2)
        with c1:
            if st.button("Clear all interactions"):
                clear_all_interactions(client, st.session_state["user_id"])
                st.success("Cleared liked/disliked history. Hidden items will be eligible again.")
                st.rerun()

    with left:
        st.markdown("### 🔎 Recommendations")
        with st.spinner("Scoring nearby restaurants..."):
            rows = recommend(
                client, embedder, conn,
                (st.session_state["lat"], st.session_state["lon"]),
                st.session_state["user_vecs"],
                st.session_state["price_pref"],
                st.session_state["weights"],
                top_k=TOP_K,
                excluded_ids=st.session_state.get("hidden_ids", set()),
            )

        if not rows:
            st.warning("No restaurants found after filtering. Try another location or clear interactions.")
        else:
            for r in rows:
                render_restaurant_card(r, user_lat=st.session_state["lat"], user_lon=st.session_state["lon"])
                b1, b2, _ = st.columns([1, 1, 8])
                with b1:
                    if st.button("👍", key=f"like_{r['idx']}"):
                        hf = apply_feedback(get_embedder(), client, st.session_state["user_id"], r, True,
                                            st.session_state["user_vecs"], st.session_state["price_pref"],
                                            st.session_state["has_feedback"])
                        st.session_state["has_feedback"] = hf
                        st.toast("Preference updated (liked). Hidden from future recommendations.")
                        st.rerun()
                with b2:
                    if st.button("👎", key=f"dislike_{r['idx']}"):
                        hf = apply_feedback(get_embedder(), client, st.session_state["user_id"], r, False,
                                            st.session_state["user_vecs"], st.session_state["price_pref"],
                                            st.session_state["has_feedback"])
                        st.session_state["has_feedback"] = hf
                        st.toast("Preference updated (disliked). Hidden from future recommendations.")
                        st.rerun()


if __name__ == "__main__":
    main()
