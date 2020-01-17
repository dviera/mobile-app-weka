package com.example.wekalm;

import android.content.res.AssetManager;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import java.io.IOException;
import java.io.InputStream;
import java.text.DecimalFormat;
import java.util.ArrayList;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;


public class MainActivity extends AppCompatActivity {
    // START
    // this is part of the project creation
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

    // END

    Button predictBtn = findViewById(R.id.predictBtn);
    predictBtn.setOnClickListener(new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            // here goes the weka model
            EditText firstNumberEditText = findViewById(R.id.firstNumEditText);
            EditText secondNumberEditText = findViewById(R.id.secondNumEditText);
            TextView resultTextView = findViewById(R.id.resultTextView);

            AssetManager assetManager = getAssets();
            InputStream is = null;
            try {
                is = assetManager.open("wekalm.model");
            } catch (IOException e) {
                e.printStackTrace();
            }

            Classifier lm = null;
            try {
                lm = (Classifier) SerializationHelper.read(is);
            } catch (Exception e) {
                e.printStackTrace();
            }


            final Attribute y = new Attribute("y");
            final Attribute x1 = new Attribute("x1");
            final Attribute x2 = new Attribute("x2");


            ArrayList<Attribute> attributeList = new ArrayList<Attribute>(){
                {
                    add(y);
                    add(x1);
                    add(x2);
                }
            };

            Instances dataUnpredicted = new Instances("TestInstances", attributeList, 1);
            dataUnpredicted.setClassIndex(0);

            final double num1 = Double.parseDouble(firstNumberEditText.getText().toString());
            final double num2 = Double.parseDouble(secondNumberEditText.getText().toString());

            DenseInstance newPrediction = new DenseInstance(dataUnpredicted.numAttributes()){
                {
                    setValue(x1, num1);
                    setValue(x2, num2);
                }
            };

            DenseInstance newInstance = newPrediction;
            newInstance.setDataset(dataUnpredicted);

            DecimalFormat df = new DecimalFormat("#.##");

            double prediction_result = 0;
            try {
                prediction_result = lm.classifyInstance(newInstance);
            } catch (Exception e) {
                e.printStackTrace();
            }
            resultTextView.setText(df.format(prediction_result) + "");













        }
    });

    }




}
