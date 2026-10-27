import io, os
os.environ["CAPTION_MOCK"] = "1"
from PIL import Image
from app import app


def _png():
    b = io.BytesIO(); Image.new("RGB", (64, 64), (10, 80, 120)).save(b, "PNG"); b.seek(0); return b


def test_caption_ok():
    r = app.test_client().post("/api/caption", data={"image": (_png(), "a.png"), "mode": "Standard Caption"})
    assert r.status_code == 200 and r.json["caption"]


def test_rejects_bad_type():
    r = app.test_client().post("/api/caption", data={"image": (io.BytesIO(b"x"), "a.gif")})
    assert r.status_code == 422


def test_no_file():
    assert app.test_client().post("/api/caption").status_code == 400
