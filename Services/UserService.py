from Models.User import User
import random
class UserService:

    def GetCategoryToServe(self, user: dict):
        categoricalPreferences = user.get("categoricalPreferences", {})
        if len(categoricalPreferences) == 0:
            return random.choice(["OddlySatisfying", "Food", "Gaming", "Cars", "Gymming"])
        maxRating = 0
        categoryToServe = None
        for category, (watched, rating) in categoricalPreferences.items():
            if rating > maxRating:
                maxRating = rating
                categoryToServe = category
        return categoryToServe
