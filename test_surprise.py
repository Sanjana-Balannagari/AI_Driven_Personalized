# test_surprise.py
from surprise import SVD, Dataset, Reader
import pandas as pd

df = pd.DataFrame({
    'user_id': [1, 1, 2],
    'food_id': [10, 20, 10],
    'rating': [5, 3, 4]
})

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'food_id', 'rating']], reader)
trainset = data.build_full_trainset()

model = SVD()
model.fit(trainset)

print("Surprise is working!")