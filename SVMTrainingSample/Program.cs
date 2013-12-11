using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using SVM; //Namespace of PSVM.dll 
using System.Diagnostics;   // for Stopwatch

namespace SVMTrainingSample
{
    class Program
    {
        static void Main(string[] args)
        {

            // Loading training and testing data
            Problem train;
            Problem test;
            try
            {
                train = Problem.Read("SampleFiles/TrainData.txt");
                test = Problem.Read("SampleFiles/TestData.txt");
            }
            catch
            {
                Console.WriteLine("Could not find the tranning and testing files");
                return;
            }

            // Initialize parameters of the kernal function
            Parameter parameters = new Parameter();
            double C = 0;
            double Gamma = 0;


            //This will do a grid-search optimization to find the best parameters, C and Gamma,
            //The search records stores to SeqentailParaOutputPath

            Console.WriteLine("Searching optimal parameters of RBF in sequential way...");

            var SequentialSW = Stopwatch.StartNew();   // Stopwatch for performance evaluation
            string SeqSearchReOutputPath = null; //
            //string SeqentailParaOutputPath = "ParaAcc_S.txt";
            ParameterSelection.Grid(train, parameters, SeqSearchReOutputPath, out C, out Gamma);
         
            SequentialSW.Stop();

            parameters.C = C;
            parameters.Gamma = Gamma;

            //This will do a grid-search optimization in parallel way to find the best parameters
            //The search records stores to ParaAcc_P.txt.

            Console.WriteLine("Searching optimal parameters of RBF in parallel way...");

            var ParallelSW = Stopwatch.StartNew();  // Stopwatch for performance evaluation
            string ParaSearchReOutputPath = null; //
            //string ParaSearchReOutputPath = "ParaAcc_P.txt";
            ParameterSelection.PGrid(train, parameters, ParaSearchReOutputPath, out C, out Gamma);
            ParallelSW.Stop();

            parameters.C = C;
            parameters.Gamma = Gamma;

            Console.WriteLine("Searching optimal parameters 2 of RBF in sequential way...");

                     
            //Output the performance comparsion
            string SequentialElapse = SequentialSW.Elapsed.TotalSeconds.ToString();
            string ParallelElapse = ParallelSW.Elapsed.TotalSeconds.ToString();
      
            Console.WriteLine(""); 
            Console.WriteLine("Sequential Grid-Search: " + SequentialElapse + " Seconds");
            Console.WriteLine("  Parallel Grid-Search: " + ParallelElapse +   " Seconds");
           
            Console.WriteLine(""); 
            //Train the model using the optimal parameters.
            Model model = Training.Train(train, parameters);
            
            //Perform classification on the test data, 
            //putting the results in results.txt.
            Prediction.Predict(test, "results.txt", model, false);
            Console.WriteLine("Press any key to finish the program...");
            Console.Read();
        
        }
    }
}
