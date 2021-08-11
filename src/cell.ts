// Type representing a node in a cellular automaton.

//import { NeuralNetwork } from './backprop'
import * as Brain from "brain.js"
import { config } from "mathjs"

// State of cell neighbourhood
type NeighbourhoodState = number[]
const MAX_HISTORY = 7
const KICKSTART_DURATION = 4

let trainingOptions: Brain.INeuralNetworkTrainingOptions = {
    iterations: 200,
    learningRate: 0.1
}


// Class representing a node in a cellular automaton.
export class Cell {
    private history: NeighbourhoodState[] 
    private _nextState: number | undefined
    private _nextHistory: NeighbourhoodState | undefined

    constructor(
        public state: number,
        public neighbours: Cell[],
        public predictor: Brain.NeuralNetwork
    ) {
        this.history = []
        this._nextHistory = undefined
        this._nextState = undefined
    }

    private getNeighbourState(): NeighbourhoodState {
        //Return an array representing the state of the neighbours
        const result: NeighbourhoodState = this.neighbours.map(neighbour => {
            return neighbour.state
        })
        return result
    }

    // Calculate the next state of the node.
    public prepareNextState() {
        //this._nextHistory = this.predict()
        //this._nextState = this._nextHistory[5]
        this._nextState = this.predictNextState()
    }

    public applyNextState() {
        if (!this._nextState == undefined) {
            throw new Error('No next state')
        }
        this.state = this._nextState ?? 0
    }

    public updateHistory() {
        this._nextHistory = this.getNeighbourState()
        this.history.push(this._nextHistory)
        if (this.history.length > MAX_HISTORY) {
            this.history.shift()
        }

    }

    public train() {
        if (this.history.length < KICKSTART_DURATION) {
            return
        }
        trainingOptions.learningRate = 0.5
        trainingOptions.iterations = 1

        // Make training set from history
        const trainingSet: {input: number[], output: number[]}[] = []
        for (let i = 0; i < this.history.length-1; i++) {
            trainingSet.push({
                input: this.history[i],
                output: [this.history[i+1][5]]
            })
        }
        trainingSet.push({
            input: this.history[this.history.length-1],
            output: [this.state]
        })
        this.predictor.train(trainingSet, trainingOptions)


    }

    // Choose a future for myself based on expectations of the past
    /*public predict(): NeighbourhoodState {
        //this.train()
        return this.predictor.predict(this.getNeighbourState())
    }*/

    // Choose a future for myself based on expectations of the past
    public predictNextState(): number {
        //this.train()
        if (this.history.length < KICKSTART_DURATION) {
            return Math.random()
        }
        const input = this.getNeighbourState()
        const output = this.predictor.run(input)
        return output[0]
    }
}
