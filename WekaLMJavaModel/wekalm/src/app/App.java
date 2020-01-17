package app;

import weka.core.converters.CSVLoader;
import weka.core.Instances;
import weka.classifiers.functions.LinearRegression;
import weka.core.SerializationHelper;

import java.io.File;

public class App {
    private static final String PATH_DATA = "./data/wekalm.csv";

    public static void main(String[] args) throws Exception {

        // load the data in CSV format
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(PATH_DATA));
        Instances data = loader.getDataSet();
        data.setClassIndex(0);

        //System.out.println(data.toSummaryString());

        // build a linear regression model
        LinearRegression lm = new LinearRegression();
        lm.buildClassifier(data);
        double[] coef = lm.coefficients();
        System.out.println(lm);
        
        /*
        for (int i = 1; i < data.numAttributes()+1; i++) {
            System.out.println(coef[i]);
            
        }
        */


        // save model
        SerializationHelper.write("assets/wekalm.model", lm);

    }
}