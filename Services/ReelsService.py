from typing import List
from Services.MongoService import MongoService
import random
class ReelsService:
    def GetInitialReels(self, initialPreferences: List[str], mongoService: MongoService):
        randomCategory = random.choice(initialPreferences)
        print("Random category: ", randomCategory)
        if randomCategory == "Oddly Satisfying":
            randomCategory = "OddlySatisfying"
        randomReel = random.choice(mongoService.GetReelsByCategory(randomCategory))
        print("Random reel: ", randomReel)
        return str(randomReel.get("_id"))
    
    def GetSerendipityReel(self, userId: str, currentReelId: str, mongoService: MongoService):
        reelCategories = ["OddlySatisfying", "Food", "Gaming"]
        user = mongoService.FindUser(userId)
        interactions = user.get("interactions", {})
        currentReel = mongoService.GetReel(currentReelId)
        category = currentReel.get("category")
        if category in reelCategories:
            print("Removing category: ", category)
            reelCategories.remove(category)
        else:
            print("DB INCONSISTENCY: Category not found in reelCategories")
        randomCategory = random.choice(reelCategories)
        categoryReels = mongoService.GetReelsByCategory(randomCategory)
        while True:
            randomReel = random.choice(categoryReels)
            if randomReel.get("_id") not in interactions:
                return str(randomReel.get("_id"))
            else:
                print("Reel already watched, trying again")
                continue

