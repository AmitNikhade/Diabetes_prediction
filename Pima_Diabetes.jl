using PyCall  # import packages
using Pkg
using Distributions, Plots
using DataFrames
using CSV
#load data
df1 = CSV.read("/home/lucifer/Downloads/archive (4)/diabetes.csv", DataFrame)
df = copy(df1)
#Remove outliers
df1 = df1[df1.Pregnancies .<14, :]
df1 = df1[df1.Glucose .> 50, :]
df1 = df1[df1.BloodPressure .< 120, :]
df1 = df1[df1.SkinThickness .< 80, :]
df1 = df1[df1.Insulin .< 600, :]
df1 = df1[df1.BMI .< 55, :]
df1 = df1[df1.DiabetesPedigreeFunction .< 2, :]
df1 = df1[df1.Age .< 70, :]

# Remove Skewness by passing the column values through the log function
Ins=[]
function remove_skew(x)
    for i in x
        if i>0||i!=0
            v = log(i)
            append!(Ins, v)
        else
            v = 0
            append!(Ins, v)
        end
    end 
end

# Remove skewness  from the distribution
remove_skew(df1[!, :Insulin])
df1[!, :Insulin] = Float64.(Ins);
# feature scaling
using StatsBase
new_columns = names(df1)
n = length(names(df1))
c_names = names(df1)
for x in range(1, stop=n, length=n)
    x = Int64.(x)
    col = Float64.(df1[:, x])
    s = fit(ZScoreTransform, col)
    colm = (c_names[x]*"_"*(string(x)))
    df1[:, colm] = StatsBase.transform!(s, col)
end
# select features
features = select(df1, Between(:Pregnancies_1, :Age_8))

# select label
target = df1[!, :Outcome];

@pyimport sklearn.model_selection as ms
# split data into train|test data
X_train, X_test, y_train, y_test = ms.train_test_split(Array(features),Array(target), test_size = 0.2, random_state=42)

@pyimport sklearn
# import sklearn
svm = pyimport("sklearn.svm") #for SVC
tree = pyimport("sklearn.tree") #for decision tree
neighbors = pyimport("sklearn.neighbors") #for kn
linear_model = pyimport("sklearn.linear_model") # for logistic regression
ensemble = pyimport("sklearn.ensemble") #for ensemble methods (ex. Randomforest, 
                                        #AdaBoost, GradientBoosting)

                                        # list of models we are going to use for classification task 
models = [svm.SVC(),tree.DecisionTreeClassifier(),neighbors.KNeighborsClassifier(),linear_model.LogisticRegression(),ensemble.RandomForestClassifier(),ensemble.AdaBoostClassifier(),ensemble.GradientBoostingClassifier()]

metrics = pyimport("sklearn.metrics")
# import metrics(acc. score)
for model in models
    #fit train & test data to every model and print the accuracy
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(model, metrics.accuracy_score(y_test,y_pred),"\n")
end
