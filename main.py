from utils import *
import draw

def main():
    # train_data, train_labels, test_data, test_labels = dataset()
    ld = Network([])
    ld.load('network.npz')
    draw.build_buttons(ld)
    draw.loop()

if __name__ == "__main__":
    main()
