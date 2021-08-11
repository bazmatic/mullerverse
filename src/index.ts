import { World } from "./world";

const world = new World(6, 6);
world.randomize()
world.run(1000)
