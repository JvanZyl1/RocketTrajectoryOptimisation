import multiprocessing

cpu_count = multiprocessing.cpu_count()
print(f"Number of CPUs: {cpu_count}")