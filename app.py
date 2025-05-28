from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List
from Models.User import User
from fastapi.responses import JSONResponse, FileResponse
from Services.MongoService import MongoService
from Services.SimilarReelFinderService import SimilarReelFinder
from dotenv import load_dotenv
from Models.LoginRequest import LoginRequest
from Models.SetInitialPreferencesRequest import SetInitialPreferencesRequest
import os
from Services.ReelsService import ReelsService
from Services.UserService import UserService
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def get_mongo_service():
    if not hasattr(app.state, "mongo_service"):
        app.state.mongo_service = MongoService()
    return app.state.mongo_service

@app.post("/api/login")
async def Login(loginRequest: LoginRequest):
    try:
        mongo_service = get_mongo_service()
        userId, existingUser = mongo_service.CreateOrGetUser(loginRequest.email, loginRequest.password)
        if existingUser:
            userService = UserService()
            reelService = ReelsService()
            user = mongo_service.FindUser(userId)
            categoryReelToServe = userService.GetCategoryToServe(user)
            reel_id = reelService.GetReelByCategory(categoryReelToServe, user.get("interactions", {}), mongo_service)
            return {"id": userId, "existingUser": existingUser, "reel_id": reel_id}
        else:
            return {"id": userId, "existingUser": existingUser}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.patch("/api/set-initial-preferences")
async def SetInitialPreferences(setInitialPreferencesRequest: SetInitialPreferencesRequest):
    try:
        mongo_service = get_mongo_service()
        mongo_service.SetInitialPreferences(setInitialPreferencesRequest.userId, setInitialPreferencesRequest.initialPreferences)
        reelsService = ReelsService()
        randomInitialReelId = reelsService.GetInitialReels(setInitialPreferencesRequest.initialPreferences, mongo_service)
        print("Initial preference reel: ", randomInitialReelId)
        return {"success": True, "initialPreferenceReelId": randomInitialReelId}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
@app.get("/")
async def root():
    return {"message": "Welcome to ReelsRS API"}

@app.get("/api/reel")
async def GetReel(userId: str, lastWatchedReelId: str):
    try:
        mongo_service = get_mongo_service()
        userService = UserService()
        reelService = ReelsService()
        user = mongo_service.FindUser(userId)
        reelMetadata = mongo_service.GetReel(lastWatchedReelId)
        categoryReelToServe = userService.GetCategoryToServe(user)
        print("Last watched reel category: ", reelMetadata.get("category"))
        print("Category to serve: ", categoryReelToServe)
        if categoryReelToServe is None or categoryReelToServe == reelMetadata.get("category"):
            print("Finding similar reel")
            finder = SimilarReelFinder(mongo_service=mongo_service)
            similar_reel_id = finder.find_similar_reel(reel_id=lastWatchedReelId, user_interactions=user.get("interactions", {}))
            return {"next_reel_id": similar_reel_id}
        else:
            print("Finding reel by category")
            return {
                "next_reel_id": reelService.GetReelByCategory(categoryReelToServe, user.get("interactions", {}), mongo_service)
                }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/reel-content")
async def GetReelContent(reelId: str):
    try:
        mongo_service = get_mongo_service()
        reel = mongo_service.GetReel(reelId)
        return FileResponse(reel.get("path"), media_type="video/mp4")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/serendipity-reel")
async def GetSerendipityReel(userId: str, currentReelId: str):
    try:
        mongo_service = get_mongo_service()
        reelsService = ReelsService()
        reel = reelsService.GetSerendipityReel(userId, currentReelId, mongo_service)
        return {"reelId": reel}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
        
@app.patch("/api/mark-as-watched")
async def MarkAsWatched(userId: str, reelId: str, rating: int):
    try:
        mongo_service = get_mongo_service()
        mongo_service.MarkAsWatched(userId, reelId, rating)
        return {"success": True}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 