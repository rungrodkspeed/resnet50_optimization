
def main():
    
    epsilon:float = 1.0
    one:float = 1.0
    
    while(one + epsilon != 1.0):
        
        epsilon /= 2
    
    print(f'epsilon : {epsilon}')

    if one + epsilon == 1.0:
        print(True)
    else: 
        print(False)




if __name__ == "__main__":
    main()