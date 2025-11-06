# app.py
from flask import Flask, render_template, request
from models.recommender import get_meal_plan

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        prefs = request.form.getlist("preferences")
        try:
            calories = int(request.form["calories"])
        except:
            calories = 2000
        # === FIXED: Pass k=5 ===
        plan, ids = get_meal_plan(prefs, calories, k=5)
        return render_template("results.html", plan=plan, calories=calories)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)