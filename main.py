from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from database import search_by_tag
from fastapi.responses import RedirectResponse
from database import delete_image_from_db
from database import get_all_profiles, rename_profile, search_by_profile

app = FastAPI()

@app.post("/delete/{image_id}")
async def delete_image(image_id: int, tag: str = None):
    delete_image_from_db(image_id)
    # Redirect back to the gallery, keeping the current search tag if it exists
    url = f"/?tag={tag}" if tag else "/"
    return RedirectResponse(url=url, status_code=303)

# Mount the static folder so browser can see your images
app.mount("/my_images", StaticFiles(directory="./my_images"), name="images")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, tag: str = None, profile_id: int = None):
    from database import get_tag_stats
    stats = get_tag_stats()
    profiles = get_all_profiles()
    
    results = []
    if tag:
        results = search_by_tag(tag)
    elif profile_id:
        # Search images of a specific person
        results = search_by_profile(profile_id)
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "results": results, 
        "tag": tag,
        "profile_id": profile_id,
        "stats": stats,
        "profiles": profiles
    })
@app.post("/rename/{profile_id}")
async def rename(profile_id: int, new_name: str = Form(...)):
    rename_profile(profile_id, new_name)
    return RedirectResponse(url="/", status_code=303)
