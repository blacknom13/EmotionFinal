import multiprocessing
import pickle


# Define a custom object
class MyObject:
    def __init__(self, data):
        self.data = data


# Define a function to be run in a separate process
def process_func(shared_mem, obj):
    new_obj=pickle.loads(obj)
    new_obj.data += 1
    byte_obj = pickle.dumps(new_obj)
    for i in range(len(byte_obj)):
        shared_mem[i] = byte_obj[i]
    print(f"Data in child process: {new_obj.data}")


array = multiprocessing.Array("b", 200)

# Create a process and pass the serialized object as an argument
if __name__ == "__main__":
    # Create an instance of the custom object
    my_obj = MyObject(10)

    # Serialize the object using pickle
    serialized_obj = pickle.dumps(my_obj)
    print(serialized_obj)

    p = multiprocessing.Process(target=process_func, args=(array, serialized_obj,))
    p.start()
    p.join()

    new_obj=[]
    # Deserialize the modified object from the child process
    for i in range(200):
        new_obj.append(array[i])
        print(array[i], sep=",", end='')
    my_obj = pickle.loads(array)
    print(f"Data in parent process: {my_obj.data}")
