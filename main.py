from sklearn import datasets
import matplotlib.pyplot as plt

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

    """
    feature - характеризующая черта предмета
    example - любой предмет
    class - подвид или группа, на которые делится предмет
    """

    # Task 2: Print out the number of features & number of examples in the iris dataset
    print('Number of features ' + str(len(iris.feature_names)))
    print('Number of examples ' + str(len(iris.target)))

    # Task 3: Print out iris dataset classes
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
    

def main():
    practice_three_one()


if __name__ == '__main__':
    main()

