import { NeuralNetwork } from "./backprop";
import * as Brain from "brain.js"
import { Cell } from "./cell";

// This class specifies a cellular automaton world. It has an array of cells, each linked to their 8 nearest neighbors.
// The world is a 2D array of cells.

export class World {
    public cells: Cell[]
    public matrix: Cell[][]

    constructor(public width: number, public height: number) {


        this.cells = []
        this.matrix = []
        for (let y = 0; y < height; y++) {
            
            const row: Cell[] = []
            for (let x = 0; x < width; x++) {
                const nn = new Brain.NeuralNetworkGPU({
                    hiddenLayers: [9,9],
                    activation: 'sigmoid',
                })
                const newCell = new Cell(0, [], nn)
                this.cells.push(newCell)
                row.push(newCell)        
            }
            this.matrix.push(row)
        }
        //Assign neighbours for each cell
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                this.cells[y * width + x].neighbours = this.getNeighbours(x, y)
            }
        }
    }


    private getNeighbours(x: number, y: number): Cell[] {
        const neighbours: Cell[] = []
        for (let i = -1; i < 2; i++) {
            for (let j = -1; j < 2; j++) {
                let newX = x + i
                let newY = y + j
                // Wrap around
                if (newX < 0) {
                    newX = this.width - 1
                } else if (newX >= this.width) {
                    newX = 0
                }
                if (newY < 0) {
                    newY = this.height - 1
                } else if (newY >= this.height) {
                    newY = 0
                }
                neighbours.push(this.matrix[newY][newX])
            }
        }
        return neighbours
    }

    public step() {
        const start = Date.now()
        //let i=0;
        const permuted = shuffle(this.cells)

        for (let i=0; i<20; i++) {
            for (const cell of permuted) {
                cell.train()
            }
        }
        for (const cell of this.cells) {
            cell.prepareNextState()
        }
        for (const cell of this.cells) {
            cell.applyNextState()
        }
        for (const cell of this.cells) {
            cell.updateHistory()
        }
        console.log(`Step took ${Date.now() - start} ms`)
    }

    public render() {
        const shadings = [' ', '░', '▒', '▓', '█']   //"  .,xxxXXX##" 
        const values = this.cells.map((cell)=>{return cell.state})
        const normalisedValues = normaliseValues(values, shadings.length-1)
        const valueGrid: number[][] = []
        // Turn flat array into a matrix
        for (let y = 0; y < this.height; y++) {
            const row: number[] = []
            for (let x = 0; x < this.width; x++) {
                row.push(normalisedValues[y * this.width + x])
            }
            valueGrid.push(row)
        }
        
        for (let y = 0; y < this.height; y++) {
            let lineStr = ""
            for (let x = 0; x < this.width; x++) {
                let val = valueGrid[y][x] // Math.floor(this.matrix[y][x].state * shadings.length)
                let pixel = shadings[Math.floor(val)]
                if (pixel === undefined) {
                    pixel = 'X'
                }
                lineStr += `${pixel}${pixel}`
            }

            console.log(lineStr)
        }
        console.log('====================')
    }

    public randomize() {
        for (const cell of this.cells) {
            cell.state = Math.random()
        }
    }

    public run(steps: number) {
        //Render every 10 steps
        this.render()
        while(true) {
            this.step()
            this.render()
        }
    }

}

// Function to return a random permutation of an array, such that each element appears once
function shuffle(array: any[]): any[] {
    let currentIndex = array.length, temporaryValue, randomIndex;
    while (0 !== currentIndex) {
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex -= 1;
        temporaryValue = array[currentIndex];
        array[currentIndex] = array[randomIndex];
        array[randomIndex] = temporaryValue;
    }
    return array;
}

function normaliseValues(values: number[], newRange: number): number[] {
    const min = Math.min(...values)
    const max = Math.max(...values)
    let range = max - min
    if (range === 0) {
        range = 1
        //throw new Error(`Cannot normalise values, all values are the same`)
    }
    const newValues = []
    for (const value of values) {
        newValues.push((value - min) / range * newRange)
    }
    return newValues
}

