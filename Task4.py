

# TASK 4: Simple Flower Prediction

print("AI Flower Prediction Project")
print("----------------------------")

sl = float(input("Enter sepal length: "))
sw = float(input("Enter sepal width: "))
pl = float(input("Enter petal length: "))
pw = float(input("Enter petal width: "))

# Simple rule-based prediction
if pl < 2.5 and pw < 1:
    result = "Iris Setosa"
elif pl < 5 and pw < 1.8:
    result = "Iris Versicolor"
else:
    result = "Iris Virginica"

print("\nPrediction Result:", result)
print("Project completed successfully!")
