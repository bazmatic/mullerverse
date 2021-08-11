import {random, multiply, dotMultiply, exp, subtract, transpose, add} from 'mathjs'

export class NeuralNetwork {

    private synapse0: number[]
    private synapse1: number[]

    constructor(
        private inputNodes: number,
        private hiddenNodes: number,
        private outputNodes: number,
        private lr: number,

        ) {

        //Generate initial synapses
        this.synapse0 = random([this.inputNodes, this.hiddenNodes], -1.0, 1.0);
        this.synapse1 = random([this.hiddenNodes, this.outputNodes], -1.0, 1.0);

    }
    activation(x: number, derivate: boolean): number {
        return sigmoid(x, derivate);
    }

    setLearningRate(lr: number) {
        this.lr = lr;
    }
    train(input: number[], target: number[], cycles?: number) {
        //if (input.length !== target.length) {
        //    throw new Error(`Input and target arrays must be the same length.`);
        //}
        for (let i = 0; i < (cycles ?? 1); i++) {
            //forward
            let input_layer = input; //input data
            let hidden_layer = multiply(input_layer, this.synapse0).map(v => this.activation(v, false)); //output of hidden layer neurons (matrix!)
            let output_layer = multiply(hidden_layer, this.synapse1).map(v => this.activation(v, false)); // output of last layer neurons (matrix!)
            
            //backward
            let output_error = subtract(target, output_layer) as number[]; //calculating error (matrix!)       
            let output_delta = dotMultiply(output_error, output_layer.map(v => this.activation(v, true))); //calculating delta (vector!)
            let hidden_error = multiply(output_delta, transpose(this.synapse1)) as number[]; //calculating of error of hidden layer neurons (matrix!)
            let hidden_delta = dotMultiply(hidden_error, hidden_layer.map(v => this.activation(v, true))); //calculating delta (vector!)
        
            //gradient descent
            this.synapse1 = (add(this.synapse1, multiply(transpose(hidden_layer), multiply(output_delta, this.lr)))) as number[];
            this.synapse0 = add(this.synapse0, multiply(transpose(input_layer), multiply(hidden_delta, this.lr))) as number[];

            /*if (i % 10000 == 0) {
                console.log(`Error: ${mean(abs(output_error))}`);
            }*/
        }
    }
    public predict(input: number[]): number[] {
        let input_layer = input;
        let hidden_layer = multiply(input_layer, this.synapse0).map(v => this.activation(v, false));
        let output_layer = multiply(hidden_layer, this.synapse1).map(v => this.activation(v, false));
        return output_layer;
    }
}

function sigmoid(x: number, derivative: boolean): number {
	let fx = 1 / (1 + exp(-x));
    if (derivative)
		return fx * (1 - fx);
	return fx;
}

