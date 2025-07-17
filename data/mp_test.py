import multiprocessing as mp

def square(x):
    return x * x

# def main():
with mp.Pool(4) as pool:
    for i in pool.imap(square, [1, 2, 3, 4, 5], chunksize=2):
        print(i)

# if __name__ == "__main__":
#     mp.freeze_support()          # harmless on non-Windows; needed for PyInstaller/Windows
#     main()