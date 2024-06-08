from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

from liver_disease.constants import APP_HOST, APP_PORT
from liver_disease.pipeline.prediction_pipeline import INDliverData, INDliverClassifier
from liver_disease.pipeline.training_pipeline import TrainPipeline

from liver_disease.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
from liver_disease.logger import logging

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.Age: Optional[str] = None
        self.Gender: Optional[str] = None
        self.Total_Bilirubin: Optional[str] = None
        self.Direct_Bilirubin: Optional[str] = None
        self.Alkaline_Phosphotase: Optional[str] = None
        self.Alamine_Aminotransferase: Optional[str] = None
        self.Aspartate_Aminotransferase: Optional[str] = None
        self.Total_Protiens: Optional[str] = None
        self.Albumin: Optional[str] = None
        self.Albumin_and_Globulin_Ratio: Optional[str] = None

    async def get_indliver_data(self):
        form = await self.request.form()
        self.Age = form.get("age") or '0'
        self.Gender = form.get("gender") or '0'
        self.Total_Bilirubin = form.get("total_bilirubin") or '0'
        self.Direct_Bilirubin = form.get("direct_bilirubin") or '0'
        self.Alkaline_Phosphotase = form.get("alkaline_phosphotase") or '0'
        self.Alamine_Aminotransferase = form.get("alamine_aminotransferase") or '0'
        self.Aspartate_Aminotransferase = form.get("aspartate_aminotransferase") or '0'
        self.Total_Protiens = form.get("total_protiens") or '0'
        self.Albumin = form.get("albumin") or '0'
        self.Albumin_and_Globulin_Ratio = form.get("albumin_and_globulin_ratio") or '0'

@app.get("/", tags=["authentication"])
async def index(request: Request):
    return templates.TemplateResponse(
            "indliver.html",{"request": request, "context": "Rendering"})

@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_indliver_data()

        indliver_data = INDliverData(
            Age= form.Age,
            Gender= form.Gender,
            Total_Bilirubin= form.Total_Bilirubin,
            Direct_Bilirubin= form.Direct_Bilirubin,
            Alkaline_Phosphotase= form.Alkaline_Phosphotase,
            Alamine_Aminotransferase= form.Alamine_Aminotransferase,
            Aspartate_Aminotransferase= form.Aspartate_Aminotransferase,
            Total_Protiens= form.Total_Protiens,
            Albumin= form.Albumin,
            Albumin_and_Globulin_Ratio= form.Albumin_and_Globulin_Ratio
        )

        indliver_df = indliver_data.get_indliver_input_data_frame()
        logging.info(f"Received DataFrame: \n{indliver_df}")

        # Fill NaNs to handle missing data and avoid unexpected categories
        indliver_df = indliver_df.fillna('Unknown')

        # Drop columns if necessary
        indliver_df = indliver_df.drop(columns=['Albumin_and_Globulin_Ratio', 'Aspartate_Aminotransferase'])

        model_predictor = INDliverClassifier()

        value = model_predictor.predict(dataframe=indliver_df)[0]

        status = "Disease detected" if value == 1 else "Disease Not-detected and patient is healthy"

        return templates.TemplateResponse(
            "indliver.html",
            {"request": request, "context": status},
        )

    except Exception as e:
        return {"status": False, "error": f"{e}"}

if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
