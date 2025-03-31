import os
import tempfile

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from langchain_core.messages import HumanMessage

from .bot.agent import get_geme
from .bot.state import State

app = FastAPI(
    title="Medical--Bot🩺",
    summary="A versatile API designed to power medical information chatbots.",
    description="""
**Key Features:**

* **Accurate Information:** Accesses credible medical databases and peer-reviewed research.
* **Natural Language Understanding:** Understands complex medical queries.
* **Symptom Checker:** Helps assess potential health conditions.
* **Medication Information:** Provides medication details.
* **Disease Information:** Comprehensive information on diseases.
* **Customizable:** Easily integrable into various platforms.

**Note:** For informational purposes only, not a substitute for professional advice.
""",
    version="0.0.1",
    contact={
        "name": "Ramy Ibrahim",
        "url": "https://www.linkedin.com/in/ramy-ibrahim-020304262/",
        "email": "ramyibrahim987@gmail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

app.mount("/static", StaticFiles(directory="src/static"), name="static")
templates = Jinja2Templates(directory="src/templates")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse("geme.html", {"request": request})


@app.post("/chat")
async def chat(
    query: str = Form(...),
    user_id: str = Form("1"),
    image: UploadFile = File(None),
):
    try:
        config = {"configurable": {"thread_id": user_id}}
        human_message = HumanMessage(content=query.strip())
        GEME = get_geme()

        if image:
            print("__ocr__")
            image_bytes = await image.read()
            suffix = os.path.splitext(image.filename)[1] if image.filename else ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(image_bytes)
                tmp_file.flush()
                image_path = tmp_file.name

            res: State = GEME.invoke(
                {"messages": human_message, "image": image_path}, config=config
            )

            os.remove(image_path)
        else:
            print("__general__")
            res: State = GEME.invoke({"messages": human_message}, config=config)

        answer = res["messages"][-1].content
        return JSONResponse(content={"answer": answer})

    except Exception as e:
        return JSONResponse(
            content={"answer": f"An error occurred: {str(e)}"},
            status_code=400,
        )


@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@app.get("/contact", response_class=HTMLResponse)
async def contact_page(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})
