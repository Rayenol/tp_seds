from fastapi import FastAPI, Form, HTTPException, Header, Path,Body, Query, File, Request, Response, UploadFile, status
# Instantiating a FastAPI object handling all API routes
app = FastAPI()

# Creating a GET endpoint at the root path
@app.get("/")
async def hello_world():
         return {"hello": "world"}
# Async method returning a JSON response autmaticallyhandled by FastAPI




#app = FastAPI()
# API that expects an integer in the last part of its path
@app.get("/users/{id}") 
async def get_user(id: int):
    return {"id": id}


 
from enum import Enum

class UserType(str, Enum):
     STANDARD = "standard" 
     ADMIN = "admin"
@app.get("/users/{type}/{id}/")
async def get_user(type: UserType, id: int):
         return {"type": type, "id": id}




@app.get("/users/{id}")
async def get_user(id: int = Path(..., ge=1)):
    return {"id": id}


@app.get("/license-plates/{license}")
async def get_license_plate(license: str = Path(...,
                            regex=r"^\d{5}-\d{3}-\d{2}$")):
    return {"license": license}



@app.get("/users")
async def get_users(page: int=1 , size: int =10):
      return{"page": page, "size": size}





@app.get("/users")
async def get_user(page: int = Query(1, gt=0), size: int = Query(10, le=100)):
    return {"page": page, "size": size}





@app.post("/users")
async def create_user(name: str = Body(...), age: int = Body(...)):
    return {"name": name, "age": age}


@app.post("/createUser")
async def create_user(name: str = Form(...),
                  age: int =  Form(...)):
                return {"name": name, "age": age}




@app.post("/files")
async def upload_file(file: bytes = File(...)):
    return {"file_size": len(file)}




@app.post("/uploadFile")
async def upload_file(file: UploadFile = File(...)):
 return {"file_name": file.filename, "content_type": file.content_type}


from typing import List

@app.post("/uploadMultipleFiles")
async def upload_multiple_files(files: List[UploadFile]=File(...)):
    return [
            {"file_name": file.filename,
"content_type": file.content_type} for file in files
]



@app.get("/getHeader")
async def get_header(user_agent: str = Header(...)):
    return {"user_agent": user_agent}


@app.get("/request")
async def get_request_object(request: Request):
    return {"path": request.url.path}


@app.get("/setCookie")
async def custom_cookie(response: Response):
                          response.set_cookie("cookie-name", "cookie-value",
                         max_age=86400)
                          return {"hello": "world"}




@app.post("/password")
async def check_password(password: str = Body(...),
                             password_confirm: str = Body(...)):
                             if password != password_confirm:
                              raise HTTPException(
                                    status.HTTP_400_BAD_REQUEST,
                                     detail="Passwords don't match.", 
                                     )
                             return {"message": "Passwords match."}

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")
@app.get("/reply")
async def home(request: Request):
 return templates.TemplateResponse("/index.html",{"request":request})



from fastapi.responses import HTMLResponse 
from fastapi.templating import Jinja2Templates 
import pandas as pd
import json

templates = Jinja2Templates(directory="templates")
@app.get("/houseprices")
async def home(request: Request):
                                 df = pd.read_csv("data/house_pricing.csv", nrows=100) 
                                 js = df.to_json(orient="records") 
                                 data=json.loads(js)
                                 return templates.TemplateResponse("/houseprices.html", {"request":request,
                                     "house_prices":data})