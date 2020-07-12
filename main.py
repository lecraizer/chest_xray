from pre_processing.preconfiguration import PreProcessor
from pre_visualization.preplotter import PreVisualizer
from model.TL_model import TransferLearningModel
from model.conv_model import ConvolutionalModel
from data_loader.xray_loader import DataLoader
from utils.plotter import Visualizer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def main():

    # Set a limit do GPU work processing and download files if needed.
    P = PreProcessor() 
    P.set_gpu_limit()
    P.download_files()


    # Load images into train, test and validation sets.
    Loader = DataLoader()   
    X_train, y_train, X_val, y_val, X_test, y_test = Loader.get_data()
    
    
    # Visualize images and quantities.
    PreV = PreVisualizer()
    PreV.image_visualizer(n = 5)
    PreV.plot_countplot(y_train) # visualize countplot and save to file

    
    # Construct, compile and train model.
#     M = ConvolutionalModel()
    M = TransferLearningModel()
    history = M.train_model(X_train, y_train, (X_val, y_val))
    pred, y_true = M.evaluate_model(X_test, y_test)
    
    # Evaluate results acquired after model fitting.
    V = Visualizer()
    V.history_results(history) # plot results into a 2d history plot 
    V.plot_confusion_matrix(y_true, pred) # plot a confusion matrix and save it into a file


if __name__ == '__main__':
    main()