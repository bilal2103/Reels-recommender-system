import os
from pymongo import MongoClient
from bson.objectid import ObjectId
from Models.User import User
from Models.Reel import Reel
from dotenv import load_dotenv
from typing import List, Dict
import re
load_dotenv()

class MongoService:

    def __init__(self):
        self.client = MongoClient(os.getenv("MONGO_URI"))
        self.db = self.client[os.getenv("MONGO_DB")]
        self.users_collection = self.db["users"]
        self.reels_collection = self.db["reels"]
        self.similarities_collection = self.db["similarities"]

    def FindUser(self, user_id: str):
        user_id = ObjectId(user_id)
        return self.users_collection.find_one({"_id": user_id})
    
    def CreateOrGetUser(self, email: str, password: str):
        user = self.users_collection.find_one({"email": email})
        if user:
            return str(user["_id"]), True
        interactions = {}
        initialPreferences = []
        inserted = self.users_collection.insert_one({
            "email": email,
            "password": password,
            "interactions": interactions,
            "initialPreferences": initialPreferences
        })
        return str(inserted.inserted_id), False
    
    def SetInitialPreferences(self, user_id: str, initialPreferences: List[str]):
        user_id = ObjectId(user_id)
        self.users_collection.update_one({"_id": user_id}, {"$set": {"initialPreferences": initialPreferences}})

    def AddReel(self, reel: Reel):
        inserted = self.reels_collection.insert_one({
            "path": reel.path,
            "category": reel.category,
            "textualEmbeddings": reel.textualEmbeddings,
            "aggregatedEmbeddings": reel.aggregatedEmbeddings,
            "videoEmbeddings": reel.videoEmbeddings
        })
        return str(inserted.inserted_id)
    
    def GetReel(self, reel_id: str):
        reel_id = ObjectId(reel_id)
        return self.reels_collection.find_one({"_id": reel_id})
        
    def GetAllReels(self):
        """Retrieve all reels from the database"""
        cursor = self.reels_collection.find({})
        reels = []
        for doc in cursor:
            # Add the MongoDB ID as id field if not already present
            if '_id' in doc and 'id' not in doc:
                doc['id'] = str(doc['_id'])
            reels.append(doc)
        return reels

    def MarkAsWatched(self, user_id: str, reel_id: str, rating: int):
        reelMetadata = self.GetReel(reel_id)
        user = self.FindUser(user_id)
        currInteractions = user.get("interactions", {})
        currInteractions[reel_id] = rating
        currCategoricalPreferences = user.get("categoricalPreferences", {})
        currentlyWatched, currentAverageRating = currCategoricalPreferences.get(reelMetadata.get("category"), (0, 0.0))
        currentlyWatched += 1
        currentAverageRating = (currentAverageRating * (currentlyWatched - 1) + rating) / currentlyWatched
        currCategoricalPreferences[reelMetadata.get("category")] = (currentlyWatched, currentAverageRating)
        self.users_collection.update_one({"_id": ObjectId(user_id)}, {"$set": {"interactions": currInteractions, "categoricalPreferences": currCategoricalPreferences}})

    def GetReelByPath(self, path: str):
        # Escape special regex characters in the path
        escaped_path = re.escape(path)
        # Find reel with case-insensitive path comparison
        print("Path to compare: ", escaped_path)
        reel = self.reels_collection.find_one({"path": {"$regex": f"^{escaped_path}$", "$options": "i"}})
        if reel:
            video_name = reel["path"].split("\\")[-1]
            return {
                "video_name": video_name,
                "category": reel["category"],
                "reel_id": str(reel.get("_id"))
            }
        return None
    
    def GetReelsByCategory(self, category: str):
        allReels = self.reels_collection.find(
            {"category": {"$regex": f"^{category}", "$options": "i"}},
            {"_id": 1, "path": 1}
        )        
        return [reel for reel in allReels]

    def StoreSimilarReels(self, reel_id: str, similarities: List[Dict[str, str]]):
        self.similarities_collection.insert_one({
            "reel_id": reel_id,
            "similarities": similarities
        })
    def GetSimilarities(self, reel_id: str):
        reel_id = ObjectId(reel_id)
        reelSimilaritiesObject = self.similarities_collection.find_one({"reel_id": reel_id})
        if reelSimilaritiesObject:
            return reelSimilaritiesObject.get("similarities", [])
        return []
    
    def DeleteUncategorizedReels(self):
        print("Printing uncategorized reels")
        uncategorized_reels = self.GetReelsByCategory("uncategorized")
        for reel in uncategorized_reels:
            print(reel)
        print(self.reels_collection.delete_many({"category": "uncategorized"}))

if __name__ == "__main__":
    mongo_service = MongoService()
    mongo_service.DeleteUncategorizedReels()