def main():
    data = np.genfromtxt('are_blue_pills_magics.csv', delimiter=',', skip_header=1, usecols=(1, 2))
    second_column = data[:, 0]
    third_column = data[:, 1]
    print("Micrograms:")
    print(second_column)
    print("\nScore:")
    print(third_column)

if __name__ == "__main__":
    main()