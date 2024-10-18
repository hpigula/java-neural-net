package com.horacy.programmingbasics;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Random;
import java.lang.Math;

/**
 * The App class demonstrates a simple neural network training and predicting process.
 * It initializes networks with different parameters, trains them on a dataset, and makes predictions.
 */
public class App {
  public static void main( String[] args ) {
    App app = new App();
    app.trainAndPredict();
  }

  /**
   * Trains and evaluates neural networks with provided data and prints the prediction results.
   * This method instantiates two different neural network configurations and trains them using
   * the same dataset. It then prints out the predictions for a series of different inputs using
   * the trained models. Additionally, the method repeats this process with two other networks
   * that have specific learning factors.
   * The method performs the following steps:
   * - Creates a list of training data (a list of lists of integers).
   * - Creates a list of expected output values for the training data.
   * - Instantiates two neural networks with different epoch sizes (50 and 100).
   * - Trains these networks with the training data.
   * - Prints the prediction results for different inputs using the two trained networks.
   * - Repeats the process with two new neural networks that include specific learning factors.
   */
  public void trainAndPredict() {
    List<List<Integer>> data = new ArrayList<>();
    data.add(Arrays.asList(115, 66));
    data.add(Arrays.asList(175, 78));
    data.add(Arrays.asList(205, 72));
    data.add(Arrays.asList(120, 67));
    List<Double> answers = Arrays.asList(1.0,0.0,0.0,1.0);  

    Network network50 = new Network(50);
    network50.train(data, answers);

    Network network100 = new Network(100);
    network100.train(data, answers);

    System.out.println();
    System.out.printf("  male, 167, 73: network500: %.10f | network1000: %.10f%n", network50.predict(167, 73), network100.predict(167, 73));
    System.out.printf("female, 105, 67: network500: %.10f | network1000: %.10f%n", network50.predict(105, 67), network100.predict(105, 67));
    System.out.printf("female, 120, 72: network500: %.10f | network1000: %.10f%n", network50.predict(120, 72), network100.predict(120, 72));
    System.out.printf("  male, 143, 67: network500: %.10f | network1000: %.10f%n", network50.predict(143, 67), network100.predict(120, 72));
    System.out.printf(" male', 130, 66: network500: %.10f | network1000: %.10f%n", network50.predict(130, 66), network100.predict(130, 66));

    System.out.print("""
                    
                    *** Second Round of the functions ***
                    """
            );
    Network network50learn1 = new Network(50, 2.0);
    network50learn1.train(data, answers);

    Network network100learn1 = new Network(100, 4.1);
    network100learn1.train(data, answers);

    System.out.println();
    System.out.printf("  male, 167, 73: network50learn1: %.10f | network100learn1: %.10f%n", network50learn1.predict(167, 73), network100learn1.predict(167, 73));
    System.out.printf("female, 105, 67: network50learn1: %.10f | network100learn1: %.10f%n", network50learn1.predict(105, 67), network100learn1.predict(105, 67));
    System.out.printf("female, 120, 72: network50learn1: %.10f | network100learn1: %.10f%n", network50learn1.predict(120, 72), network100learn1.predict(120, 72));
    System.out.printf("  male, 143, 67: network50learn1: %.10f | network100learn1: %.10f%n", network50learn1.predict(143, 67), network100learn1.predict(120, 72));
    System.out.printf(" male', 130, 66: network50learn1: %.10f | network100learn1: %.10f%n", network50learn1.predict(130, 66), network100learn1.predict(130, 66));

  }
 

  static class Network {
    int epochs = 0;
    Double learnFactor = null;
    /**
     * List of neurons in the neural network. The neurons are instances of the Neuron class
     * and are used to perform computations within the network.
     */
    List<Neuron> neurons = Arrays.asList(
            new Neuron(), new Neuron(), new Neuron(),
            new Neuron(), new Neuron(),
            new Neuron());
    
    public Network(int epochs){
      this.epochs = epochs;
    }
    public Network(int epochs, Double learnFactor) {
      this.epochs = epochs;
      this.learnFactor = learnFactor;
    }

    public Double predict(Integer input1, Integer input2){
      return neurons.get(5).compute(
        neurons.get(4).compute(
	  neurons.get(2).compute(input1, input2),
	  neurons.get(1).compute(input1, input2)
	),
	neurons.get(3).compute(
	  neurons.get(1).compute(input1, input2),
	  neurons.get(0).compute(input1, input2)
	)
      );
    }
    /**
     * Trains the neural network using the provided data and corresponding answers.
     * The training process adapts the network's neurons across multiple epochs to
     * minimize the loss function.
     *
     * @param data a list of input data, where each input is a list containing two integer values
     * @param answers a list of correct output values corresponding to the input data
     */
    public void train(List<List<Integer>> data, List<Double> answers){
      Double bestEpochLoss = null;
      for (int epoch = 0; epoch < epochs; epoch++){
        // adapt neuron
        Neuron epochNeuron = neurons.get(epoch % 6);
	epochNeuron.mutate(this.learnFactor);

	List<Double> predictions;
          predictions = new ArrayList<>();
          for (int i = 0; i < data.size(); i++){
          predictions.add(i, this.predict(data.get(i).get(0), data.get(i).get(1)));
	}
        Double thisEpochLoss = Util.meanSquareLoss(answers, predictions);

	if (epoch % 10 == 0) System.out.printf("Epoch: %s | bestEpochLoss: %.15f | thisEpochLoss: %.15f%n", epoch, bestEpochLoss, thisEpochLoss);

	if (bestEpochLoss == null){
          bestEpochLoss = thisEpochLoss;
	  epochNeuron.remember();
	} else {
	  if (thisEpochLoss < bestEpochLoss){
	    bestEpochLoss = thisEpochLoss;
	    epochNeuron.remember();
	  } else {
            epochNeuron.forget();
          }
	}
      }
    }
  }

  /**
   * Represents a neuron in a neural network. This class manages the neuron's
   * weights and biases, allows for mutation of its parameters, and computes
   * the neuron's output based on input values.
   */
  static class Neuron {
    Random random = new Random();
    private Double oldBias = random.nextDouble(-1, 1), bias = random.nextDouble(-1, 1); 
    public Double oldWeight1 = random.nextDouble(-1, 1), weight1 = random.nextDouble(-1, 1); 
    private Double oldWeight2 = random.nextDouble(-1, 1), weight2 = random.nextDouble(-1, 1);
   
    /**
     * Returns a string representation of the Neuron object. The representation includes
     * the current and previous values of the neuron's bias and weights.
     *
     * @return a formatted string containing the old and current values of bias, weight1, and weight2.
     */
    public String toString(){
      return String.format("oldBias: %.15f | bias: %.15f | oldWeight1: %.15f | weight1: %.15f | oldWeight2: %.15f | weight2: %.15f", this.oldBias, this.bias, this.oldWeight1, this.weight1, this.oldWeight2, this.weight2);
    }

    /**
     * Mutates the neuron's parameters (bias, weight1, or weight2) by applying a
     * randomly calculated change factor. The change factor is influenced by the
     * provided learnFactor.
     *
     * @param learnFactor the factor influencing the magnitude of the mutation.
     *                    If null, a random value between -1 and 1 is used.
     */
    public void mutate(Double learnFactor){
      int propertyToChange = random.nextInt(0, 3);
      double changeFactor = (learnFactor == null) ? random.nextDouble(-1, 1) : (learnFactor * random.nextDouble(-1, 1));
      if (propertyToChange == 0){ 
        this.bias += changeFactor; 
      } else if (propertyToChange == 1){ 
	this.weight1 += changeFactor; 
      } else { 
	this.weight2 += changeFactor; 
      }
    }
    /**
     * Resets the neuron's bias, weight1, and weight2 to their previously stored values.
     *
     * This method is useful in scenarios where a proposed modification to the neuron's parameters
     * needs to be reverted, for example, if the modification did not lead to an improvement
     * in the neuron's performance.
     */
    public void forget(){
      bias = oldBias;
      weight1 = oldWeight1;
      weight2 = oldWeight2;
    }
    /**
     * Updates the stored previous values of bias, weight1, and weight2 to the current values.
     *
     * The method is useful for keeping track of the current state of the neuron's parameters
     * before they are potentially altered by other operations such as mutation.
     */
    public void remember(){
      oldBias = bias;
      oldWeight1 = weight1;
      oldWeight2 = weight2;
    }
    /**
     * Computes the output of the neuron given two input values.
     *
     * @param input1 the first input value
     * @param input2 the second input value
     * @return the computed output after applying the sigmoid function
     */
    public double compute(double input1, double input2){
//      this.input1 = input1;  this.input2 = input2;
      double preActivation = (this.weight1 * input1) + (this.weight2 * input2) + this.bias;
      double output = Util.sigmoid(preActivation);
      return output;
    }
  }

  static class Util {
    /**
     * Computes the sigmoid function for a given input.
     *
     * @param in the input value for which the sigmoid function is to be computed
     * @return the output of the sigmoid function
     */
    public static double sigmoid(double in){
      return 1 / (1 + Math.exp(-in));
    }
    public static double sigmoidDeriv(double in){
      double sigmoid = Util.sigmoid(in);
      return sigmoid * (1 - in);
    }
    /** Assumes array args are same length */
    public static Double meanSquareLoss(List<Double> correctAnswers, List<Double> predictedAnswers){
      double sumSquare = 0;
      for (int i = 0; i < correctAnswers.size(); i++){
        double error = correctAnswers.get(i) - predictedAnswers.get(i);
	sumSquare += (error * error);
      }
      return sumSquare / (correctAnswers.size());
    }
  }
}
