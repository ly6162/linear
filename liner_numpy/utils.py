import numpy as np
import matplotlib.pyplot as plt
import numpy as np
def datagen():
    def f(x):
        return x * 2

    # Create the targets t with some gaussian noise
    noise_variance = 0.2  # Variance of the gaussian noise
    # Gaussian noise error for each sample in x
    #x = np.random.uniform(0, 1, 10)
    x = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    noise = np.random.randn(x.shape[0]) * noise_variance

    t = f(x) + noise
    np.savetxt('data_train.txt', x)
    np.savetxt('data_teacher.txt', t)

    plt.plot(x, t, 'bo', label='t')
    # Plot the initial line
    #plt.plot([0, 1], [f(0), f(1)], 'b-', label='f(x)')
    plt.xlabel('$x$', fontsize=15)
    plt.ylabel('$t$', fontsize=15)
    plt.ylim([0, 2])
    plt.title('inputs (x) vs targets (t)')
    plt.grid()
    plt.legend(loc=2)
    plt.show()

def main():
    datagen()


if __name__ == "__main__":
    main()