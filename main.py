from sklearn import datasets, linear_model, preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def practice_three_one():
    """
        https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset
        Множество Ирисов Фишера
        1. длина наружной доли околоцветника (sepal length)
        2. ширина наружной доли околоцветника (sepal width)
        3. длина внутренней доли околоцветника (petal length)
        4. ширина внутренней доли околоцветника (petal width)
        """

    # Task 1: Use scikit-learn to load iris dataset.
    iris = datasets.load_iris()

    # Task 2: Print out the number of features & number of examples in the iris dataset

    """
    feature - характеризующая черта предмета
    example - любой предмет
    """

    print('Number of features ' + str(len(iris.feature_names)))
    print('Number of examples ' + str(len(iris.target)))

    # Task 3: Print out iris dataset classes

    """
    class - подвид или группа, на которые делится предмет
    """

    print('Number of classes ' + str(len(iris.target_names)))

    # Task 4: Display Sepal width and Sepal length features using matplotlib library.
    plt.figure(figsize=(8, 6))
    plt.grid()
    sepal_length = iris.data[:, 0]
    sepal_width = iris.data[:, 1]
    classes = iris.target
    # Обозначения цветов
    formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
    plt.scatter(sepal_length, sepal_width, c=classes)
    plt.colorbar(ticks=[0, 1, 2], format=formatter)
    # Sepal Length
    plt.xlabel(iris.feature_names[0])
    # Sepal Width
    plt.ylabel(iris.feature_names[1])
    plt.title('Visualization of Dataset')
    plt.show()

    # Task 5: Split your dataset randomly in half: training and testing. Print out number of
    # examples in your training and testing sets
    sepal_length_train, \
        sepal_length_test, \
        sepal_width_train, \
        sepal_width_test = train_test_split(sepal_length, sepal_width, train_size=8, test_size=2)
    formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

    plt.scatter(sepal_length_train, sepal_width_train)
    plt.colorbar(ticks=[0, 1, 2], format=formatter)
    # Sepal Length
    plt.xlabel(iris.feature_names[0])
    # Sepal Width
    plt.ylabel(iris.feature_names[1])
    plt.title('Visualization of Dataset')
    # Стандартизирую данные
    scaler = StandardScaler()
    scaler.fit(sepal_length_train.reshape(-1, 1))

    sepal_length_train = scaler.transform(sepal_length_train.reshape(-1, 1))
    sepal_length_test = scaler.transform(sepal_length_test.reshape(-1, 1))

    # Преобразую Continious в Categorical
    lab = preprocessing.LabelEncoder()

    sepal_width_train = lab.fit_transform(sepal_width_train)
    sepal_width_test = lab.fit_transform(sepal_width_test)

    # Классифицирую данные для тренировки
    classifier = KNeighborsClassifier(n_neighbors=2)
    classifier.fit(sepal_length_train, sepal_width_train)

    # предугадать ширину с помощью длины
    sepal_width_predict = classifier.predict(sepal_length_test)
    plt.show()
   # print(confusion_matrix(sepal_width_test, sepal_width_predict))
    print('sepal_length_train: ')
    print(sepal_length_train)
    print('')
    print(' sepal_length_test: ')
    print(sepal_length_test)
    print('')
    print('sepal_width_train: ')
    print(sepal_width_train)
    print('')
    print(' sepal_width_test: ')
    print(sepal_width_test)
    print('')


def practice_three_two():
    df = pd.read_excel('price1.xlsx')
    plt.scatter( df['area'], df['price'], color='red', marker='^')
    plt.xlabel('площадь (кв.м.)')
    plt.ylabel('стоимость (млн.руб)')
    # регрессионная модель
    reg = linear_model.LinearRegression()
    reg.fit(df[['area']], df.price)
    plt.plot(df.area, reg.predict(df[['area']]))
    plt.show()
    # предсказывание
    pred = pd.read_excel('prediction_price.xlsx')
    pred.head(3)
    p = reg.predict(pred)
    pred['predicted prices'] = p
    print("Процент успеха: " + str(reg.score(df[['area']], df.price)))

def main():
    practice_three_one()
    practice_three_two()


if __name__ == '__main__':
    main()
