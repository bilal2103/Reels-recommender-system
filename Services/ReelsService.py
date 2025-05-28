from typing import List
from Services.MongoService import MongoService
import random
class ReelsService:
    def GetInitialReels(self, initialPreferences: List[str], mongoService: MongoService):
        # Try each preference until we find one with available reels
        random.shuffle(initialPreferences)  # Randomize the order
        for randomCategory in initialPreferences:
            print("Trying category: ", randomCategory)
            if randomCategory == "Oddly Satisfying":
                randomCategory = "OddlySatisfying"
            categoryReels = mongoService.GetReelsByCategory(randomCategory)
            if categoryReels:  # If we found reels in this category
                randomReel = random.choice(categoryReels)
                print("Random reel: ", randomReel)
                return str(randomReel.get("_id"))
        
        # If no reels found in any preferred categories, try all categories
        allCategories = ["OddlySatisfying", "Food", "Gaming", "Cars", "Gym"]
        for category in allCategories:
            if category not in [p.replace("Oddly Satisfying", "OddlySatisfying") for p in initialPreferences]:
                categoryReels = mongoService.GetReelsByCategory(category)
                if categoryReels:
                    randomReel = random.choice(categoryReels)
                    print("Found reel in non-preferred category: ", category)
                    return str(randomReel.get("_id"))
        
        print("No reels found in any category")
        return None
    
    def GetSerendipityReel(self, userId: str, currentReelId: str, mongoService: MongoService):
        reelCategories = ["OddlySatisfying", "Food", "Gaming", "Cars", "Gym"]
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
        print("Getting reels for category: ", randomCategory)
        categoryReels = mongoService.GetReelsByCategory(randomCategory)
        while True:
            randomReel = random.choice(categoryReels)
            if randomReel.get("_id") not in interactions:
                return str(randomReel.get("_id"))
            else:
                print("Reel already watched, trying again")
                continue
    
    def GetReelByCategory(self, category: str, interactions: dict, mongoService: MongoService):
        categoryReels = mongoService.GetReelsByCategory(category)
        for reel in categoryReels:
            if reel.get("_id") not in interactions:
                return str(reel.get("_id"))
        print("No reel found in category: ", category)
        allCategories = ["OddlySatisfying", "Food", "Gaming", "Cars", "Gym"]
        allCategories.remove(category)
        for category in allCategories:
            categoryReels = mongoService.GetReelsByCategory(category)
            for reel in categoryReels:
                if reel.get("_id") not in interactions:
                    return str(reel.get("_id"))
        print("No reel found in any category")
        return None

