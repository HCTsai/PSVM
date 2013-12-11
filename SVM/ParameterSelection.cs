/*
 * SVM.NET Library
 * Copyright (C) 2008 Matthew Johnson
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * SVM.NET Library with Parallel Improvments
 * Copyright (C) 2013 Hsiao-Chien Tsai
 * 
 * All source codes are based on the original author, Matther Johnson.
 * The original functions with prefix P are the parallel version modified by Hsiao-Chien Tsai 
 * 
 */

using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using System.Threading;

namespace SVM
{
    /// <summary>
    /// This class contains routines which perform parameter selection for a model which uses C-SVC and
    /// an RBF kernel.
    /// </summary>
    public static class ParameterSelection
    {
        /// <summary>
        /// Default number of times to divide the data.
        /// </summary>
        public const int NFOLD = 5;
        /// <summary>
        /// Default minimum power of 2 for the C value (-5)
        /// </summary>
        public const int MIN_C = -5;
        /// <summary>
        /// Default maximum power of 2 for the C value (15)
        /// </summary>
        public const int MAX_C = 15;
        /// <summary>
        /// Default power iteration step for the C value (2)
        /// </summary>
        public const int C_STEP = 2;
        /// <summary>
        /// Default minimum power of 2 for the Gamma value (-15)
        /// </summary>
        public const int MIN_G = -15;
        /// <summary>
        /// Default maximum power of 2 for the Gamma Value (3)
        /// </summary>
        public const int MAX_G = 10;
        /// <summary>
        /// Default power iteration step for the Gamma value (2)
        /// </summary>
        public const int G_STEP = 2;

        /// <summary>
        /// Returns a logarithmic list of values from minimum power of 2 to the maximum power of 2 using the provided iteration size.
        /// </summary>
        /// <param name="minPower">The minimum power of 2</param>
        /// <param name="maxPower">The maximum power of 2</param>
        /// <param name="iteration">The iteration size to use in powers</param>
        /// <returns></returns>
        public static List<double> GetList(double minPower, double maxPower, double iteration)
        {
            List<double> list = new List<double>();
            for (double d = minPower; d <= maxPower; d += iteration)
                list.Add(Math.Pow(2, d));
            return list;
        }

        /// <summary>
        /// Performs a Grid parameter selection, trying all possible combinations of the two lists and returning the
        /// combination which performed best.  The default ranges of C and Gamma values are used.  Use this method if there is no validation data available, and it will
        /// divide it 5 times to allow 5-fold validation (training on 4/5 and validating on 1/5, 5 times).
        /// </summary>
        /// <param name="problem">The training data</param>
        /// <param name="parameters">The parameters to use when optimizing</param>
        /// <param name="outputFile">Output file for the parameter results.</param>
        /// <param name="C">The optimal C value will be put into this variable</param>
        /// <param name="Gamma">The optimal Gamma value will be put into this variable</param>
        public static void Grid(
            Problem problem,
            Parameter parameters,
            string outputFile,
            out double C,
            out double Gamma)
        {
            Grid(problem, parameters, GetList(MIN_C, MAX_C, C_STEP), GetList(MIN_G, MAX_G, G_STEP), outputFile, NFOLD, out C, out Gamma);
        }
        /// <summary>
        /// Performs a Grid parameter selection, trying all possible combinations of the two lists and returning the
        /// combination which performed best.  Use this method if there is no validation data available, and it will
        /// divide it 5 times to allow 5-fold validation (training on 4/5 and validating on 1/5, 5 times).
        /// </summary>
        /// <param name="problem">The training data</param>
        /// <param name="parameters">The parameters to use when optimizing</param>
        /// <param name="CValues">The set of C values to use</param>
        /// <param name="GammaValues">The set of Gamma values to use</param>
        /// <param name="outputFile">Output file for the parameter results.</param>
        /// <param name="C">The optimal C value will be put into this variable</param>
        /// <param name="Gamma">The optimal Gamma value will be put into this variable</param>
        public static void Grid(
            Problem problem,
            Parameter parameters,
            List<double> CValues,
            List<double> GammaValues,
            string outputFile,
            out double C,
            out double Gamma)
        {
            Grid(problem, parameters, CValues, GammaValues, outputFile, NFOLD, out C, out Gamma);
        }
        /// <summary>
        /// Performs a Grid parameter selection, trying all possible combinations of the two lists and returning the
        /// combination which performed best.  Use this method if validation data isn't available, as it will
        /// divide the training data and train on a portion of it and test on the rest.
        /// </summary>
        /// <param name="problem">The training data</param>
        /// <param name="parameters">The parameters to use when optimizing</param>
        /// <param name="CValues">The set of C values to use</param>
        /// <param name="GammaValues">The set of Gamma values to use</param>
        /// <param name="outputFile">Output file for the parameter results.</param>
        /// <param name="nrfold">The number of times the data should be divided for validation</param>
        /// <param name="C">The optimal C value will be placed in this variable</param>
        /// <param name="Gamma">The optimal Gamma value will be placed in this variable</param>
        public static void Grid(
            Problem problem,
            Parameter parameters,
            List<double> CValues, 
            List<double> GammaValues, 
            string outputFile,
            int nrfold,
            out double C,
            out double Gamma)
        {
            C = 0;
            Gamma = 0;
            double crossValidation = double.MinValue;
            StreamWriter output = null;
            if(outputFile != null)
                output = new StreamWriter(outputFile);
            for(int i=0; i<CValues.Count; i++)
                for (int j = 0; j < GammaValues.Count; j++)
                {
                    parameters.C = CValues[i];
                    parameters.Gamma = GammaValues[j];
                    double test = Training.PerformCrossValidation(problem, parameters, nrfold);
                  //  Console.Write("{0} {1} {2}", parameters.C, parameters.Gamma, test);
                    if(output != null)
                        output.WriteLine("{0} {1} {2}", parameters.C, parameters.Gamma, test);
                    if (test > crossValidation)
                    {
                        C = parameters.C;
                        Gamma = parameters.Gamma;
                        crossValidation = test;
                   //     Console.WriteLine(" New Maximum!");
                    }
                  //  else Console.WriteLine();
                }
            if(output != null)
                output.Close();
        }
        /// <summary>
        /// Performs a Grid parameter selection, trying all possible combinations of the two lists and returning the
        /// combination which performed best.  Uses the default values of C and Gamma.
        /// </summary>
        /// <param name="problem">The training data</param>
        /// <param name="validation">The validation data</param>
        /// <param name="parameters">The parameters to use when optimizing</param>
        /// <param name="outputFile">The output file for the parameter results</param>
        /// <param name="C">The optimal C value will be placed in this variable</param>
        /// <param name="Gamma">The optimal Gamma value will be placed in this variable</param>
        public static void Grid(
            Problem problem,
            Problem validation,
            Parameter parameters,
            string outputFile,
            out double C,
            out double Gamma)
        {
            Grid(problem, validation, parameters, GetList(MIN_C, MAX_C, C_STEP), GetList(MIN_G, MAX_G, G_STEP), outputFile, out C, out Gamma);
        }
        /// <summary>
        /// Performs a Grid parameter selection, trying all possible combinations of the two lists and returning the
        /// combination which performed best.
        /// </summary>
        /// <param name="problem">The training data</param>
        /// <param name="validation">The validation data</param>
        /// <param name="parameters">The parameters to use when optimizing</param>
        /// <param name="CValues">The C values to use</param>
        /// <param name="GammaValues">The Gamma values to use</param>
        /// <param name="outputFile">The output file for the parameter results</param>
        /// <param name="C">The optimal C value will be placed in this variable</param>
        /// <param name="Gamma">The optimal Gamma value will be placed in this variable</param>
        public static void Grid(
            Problem problem,
            Problem validation,
            Parameter parameters,
            List<double> CValues,
            List<double> GammaValues,
            string outputFile,
            out double C,
            out double Gamma)
        {
            C = 0;
            Gamma = 0;
            double maxScore = double.MinValue;
            StreamWriter output = null;
            if(outputFile != null)
                output = new StreamWriter(outputFile);
            for (int i = 0; i < CValues.Count; i++)
                for (int j = 0; j < GammaValues.Count; j++)
                {
                    parameters.C = CValues[i];
                    parameters.Gamma = GammaValues[j];

                    Model model = Training.Train(problem, parameters);
                    double test = Prediction.Predict(validation, "tmp.txt", model, false);
                //    Console.Write("{0} {1} {2}", parameters.C, parameters.Gamma, test);
                    if(output != null)
                        output.WriteLine("{0} {1} {2}", parameters.C, parameters.Gamma, test);
                    if (test > maxScore)
                    {
                        C = parameters.C;
                        Gamma = parameters.Gamma;
                        maxScore = test;
               //         Console.WriteLine(" New Maximum!");
                    }
                //    else Console.WriteLine();
                }
            if(output != null)
                output.Close();
        }
        //

        /// <summary>
        /// Performs a Grid parameter selection in parallel way, trying all possible combinations of the two lists and returning the
        /// combination which performed best. Use this method if there is no validation data available.
        /// </summary>
        /// <param name="problem">The training data</param>
        /// <param name="parameters">The parameters C and Gamma to use when optimizing</param>
        /// <param name="outputFile">Output file for the parameter results.</param>
        /// <param name="C">The optimal C value will be put into this variable</param>
        /// <param name="Gamma">The optimal Gamma value will be put into this variable</param>
        public static void PGrid(Problem problem, Parameter parameters, string outputFile,
            out double C, out double Gamma)
        {
            PGrid(problem, parameters, GetList(MIN_C, MAX_C, C_STEP), GetList(MIN_G, MAX_G, G_STEP), outputFile, NFOLD, out C, out Gamma);
        }
        /// <summary>
        /// Performs a Grid parameter selection in parallel way, trying all possible combinations of the two lists and returning the
        /// combination which performed best.  Use this method if there is no validation data available.
        /// </summary>
        /// <param name="problem">The training data</param>
        /// <param name="parameters">The parameters to use when optimizing</param>
        /// <param name="CValues">The set of C values to use</param>
        /// <param name="GammaValues">The set of Gamma values to use</param>
        /// <param name="outputFile">Output file for the parameter results.</param>
        /// <param name="nrfold">The number of times the data should be divided for validation</param>
        /// <param name="C">The optimal C value will be put into this variable</param>
        /// <param name="Gamma">The optimal Gamma value will be put into this variable</param>
        public static void PGrid(Problem problem, Parameter parameters, List<double> CValues, List<double> GammaValues,
           string outputFile, int nrfold, out double C, out double Gamma)
        {
            C = 0;
            Gamma = 0;

            double BestC = 0;
            double BestGamma = 0;
            object sync = new object();
            double GlobalMaxAccuracy = 0;

            StreamWriter output = null;
            if (outputFile != null)
                output = new StreamWriter(outputFile);


            Parallel.ForEach(ParameterListToEnumerable(CValues,GammaValues),
                () => new RBFParameter(),
                (RBFParameter CurrPara, ParallelLoopState pls, RBFParameter LocalPara) =>
                {
                    // Time-consuming works process in local threads.  
                     Parameter pa = new Parameter();
                     pa.C = CurrPara.c;
                     pa.Gamma= CurrPara.gamma;
                     CurrPara.accuracy = Training.PerformCrossValidation(problem, pa, nrfold);
                     return LocalPara = CurrPara;
                },
                (RBFParameter LocalPara) =>
                {

                     //Acesses shared variable in main thread
                    lock (sync)
                    {
                        if (output != null)
                            output.WriteLine("{0} {1} {2}", LocalPara.c, LocalPara.gamma, LocalPara.accuracy);

                        if (LocalPara.accuracy > GlobalMaxAccuracy)
                        {
                            // GlobalTotal = localTotal;
                            BestC = LocalPara.c;
                            BestGamma = LocalPara.gamma;
                            GlobalMaxAccuracy = LocalPara.accuracy;
                        }
                    }
                });

            // Update Outputs to Best Values
            C = BestC;
            Gamma = BestGamma;

            if (output != null)
                output.Close();
        }

        /// <summary>
        /// Performs a Grid parameter selection in parallel way, trying all possible combinations of the two lists and returning the
        /// combination which performed best. 
        /// </summary>
        /// <param name="problem">The training data</param>
        /// <param name="validation">The validation data</param>
        /// <param name="parameters">The parameters to use when optimizing</param>
        /// <param name="outputFile">The output file for the parameter results</param>
        /// <param name="C">The optimal C value will be placed in this variable</param>
        /// <param name="Gamma">The optimal Gamma value will be placed in this variable</param>
        public static void PGrid( Problem problem, Problem validation, Parameter parameters, string outputFile,
           out double C, out double Gamma)
        {
            PGrid(problem, validation, parameters, GetList(MIN_C, MAX_C, C_STEP), GetList(MIN_G, MAX_G, G_STEP), outputFile, out C, out Gamma);
        }
        /// <summary>
        /// Performs a Grid parameter selection, trying all possible combinations of the two lists and returning the
        /// combination which performed best.
        /// </summary>
        /// <param name="problem">The training data</param>
        /// <param name="validation">The validation data</param>
        /// <param name="parameters">The parameters to use when optimizing</param>
        /// <param name="CValues">The C values to use</param>
        /// <param name="GammaValues">The Gamma values to use</param>
        /// <param name="outputFile">The output file for the parameter results</param>
        /// <param name="C">The optimal C value will be placed in this variable</param>
        /// <param name="Gamma">The optimal Gamma value will be placed in this variable</param>
        public static void PGrid(Problem problem, Problem validation, Parameter parameters,
            List<double> CValues, List<double> GammaValues, string outputFile,  out double C, out double Gamma)
        {
            C = 0;
            Gamma = 0;

            double BestC = 0;
            double BestGamma = 0;
            object sync = new object();
            double GlobalMaxAccuracy = 0;
           
            StreamWriter output = null;
            if (outputFile != null)
                output = new StreamWriter(outputFile);

            Parallel.ForEach(ParameterListToEnumerable(CValues, GammaValues),
              () => new RBFParameter(),
              (RBFParameter CurrPara, ParallelLoopState pls, RBFParameter LocalPara) =>
              {
                  // Time-consuming works process in local threads.  
                  Parameter pa = new Parameter();
                  pa.C = CurrPara.c;
                  pa.Gamma = CurrPara.gamma;
                  Model model = Training.Train(problem, pa);
                  //Using Thread.CurrentThread.ManagedThreadId prevents File I/O interlocks. 
                  string TempFileName = "TempPredictOut_" + Thread.CurrentThread.ManagedThreadId.ToString();
                  CurrPara.accuracy = Prediction.Predict(validation, TempFileName, model, false);
                  return LocalPara = CurrPara;
              },
              (RBFParameter LocalPara) =>
              {

                  //Acesses shared variable in main thread
                  lock (sync)
                  {
                      if (output != null)
                          output.WriteLine("{0} {1} {2}", LocalPara.c, LocalPara.gamma, LocalPara.accuracy);

                      if (LocalPara.accuracy > GlobalMaxAccuracy)
                      {
                          // GlobalTotal = localTotal;
                          BestC = LocalPara.c;
                          BestGamma = LocalPara.gamma;
                          GlobalMaxAccuracy = LocalPara.accuracy;
                      }
                  }
              });

            // Update Outputs to Best Values
            C = BestC;
            Gamma = BestGamma;

            if (output != null)
                output.Close();
        }


        public static IEnumerable<RBFParameter> ParameterListToEnumerable(List<double> cList, List<double> gList)
        {
           
            for (int i = 0; i < cList.Count; i++)
            {
                for (int j = 0; j < gList.Count; j++)
                {
                    RBFParameter p = new RBFParameter(cList[i], gList[j]);
                    yield return p;
                }
            }



        }

        public static IEnumerable<List<RBFParameter>> ParaListToEnumerableChunk(List<double> cList, List<double> gList)
        {
            List<RBFParameter> RetList = new List<RBFParameter>();
            int ChunkCount = 4;
            for (int i = 0; i < cList.Count; i++)
            {
                for (int j = 0; j < gList.Count; j++)
                {
                    RBFParameter p = new RBFParameter(cList[i], gList[j]);
                    RetList.Add(p);
                    if (RetList.Count >= ChunkCount)
                    {
                        yield return RetList;
                        RetList.Clear();
                    }
                }
            }



        }

        public static void PCGrid(Problem problem, Parameter parameters, string outputFile, out double C, out double Gamma)
        {
            PCGrid(problem, parameters, GetList(MIN_C, MAX_C, C_STEP), GetList(MIN_G, MAX_G, G_STEP), outputFile, NFOLD, out C, out Gamma);
        }
        public static void PCGrid(Problem problem, Parameter parameters, List<double> CValues, List<double> GammaValues,
           string outputFile, int nrfold, out double C, out double Gamma)
        {
            C = 0;
            Gamma = 0;

            double BestC = 0;
            double BestGamma = 0;
            object sync = new object();
            double GlobalMaxAccuracy = 0;

            StreamWriter output = null;
            if (outputFile != null)
                output = new StreamWriter(outputFile);


            Parallel.ForEach(ParaListToEnumerableChunk(CValues, GammaValues),
                () => new RBFParameter(),  //initial value for the local variable
                (List<RBFParameter> LocalParaList, ParallelLoopState pls, RBFParameter LocalMaxPara) =>
                {
                    int LocalCount = LocalParaList.Count;
                    // Time-consuming works process in local threads.  
                    Parameter pa = new Parameter();
                    LocalMaxPara = LocalParaList[0];

                    for (int i = 1; i < LocalParaList.Count; i++)
                    {

                        int index = i;
                        if (index < LocalCount)
                        {
                            pa.C = LocalParaList[index].c;
                            pa.Gamma = LocalParaList[index].gamma;
                            LocalParaList[index].accuracy = Training.PerformCrossValidation(problem, pa, nrfold);
                            if (LocalParaList[index].accuracy > LocalMaxPara.accuracy)
                            {
                                LocalMaxPara = LocalParaList[index];
                            }
                        }

                    }
                    return LocalMaxPara;
                },
                (RBFParameter LocalMaxPara) =>
                {
                    if (LocalMaxPara != null)
                    {
                        //Acesses shared variable in main thread
                        lock (sync)
                        {
                            if (output != null)
                                output.WriteLine("{0} {1} {2}", LocalMaxPara.c, LocalMaxPara.gamma, LocalMaxPara.accuracy);

                            if (LocalMaxPara.accuracy > GlobalMaxAccuracy)
                            {
                                // GlobalTotal = localTotal;
                                BestC = LocalMaxPara.c;
                                BestGamma = LocalMaxPara.gamma;
                                GlobalMaxAccuracy = LocalMaxPara.accuracy;
                            }
                        }
                    }
                });

            // Update Outputs to Best Values
            C = BestC;
            Gamma = BestGamma;

            if (output != null)
                output.Close();
        }
    }
    
    /// <summary>
    /// Class to store RBF Parameters and its accuracy
    /// </summary>
    public class RBFParameter
    {

        public double c { set; get; }
        public double gamma { set; get; }
        public double accuracy { set; get; }
        public RBFParameter(double ci, double gi, double ac = 0)
        {
            c = ci;
            gamma = gi;
            accuracy = ac;
        }
        public RBFParameter()
        {

            c = 0;
            gamma = 0;
            accuracy = 0;
        }
    }
}
