from VisualPyGame import Visualizer


if __name__ == "__main__":
    methods = Visualizer.getMethods()
    print("Available sorters: ", ", ".join(methods), sep="")

    method = input("Sorter: ")
    while (method not in methods):
        print("Invalid")
        inp = input("Sorter: ")
    display = Visualizer(win_dims=(1200, 400))
    display.run(method=method, n=600, speed=1000)
