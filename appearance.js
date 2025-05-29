import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Fill-ChatterBox.appearance", // Extension name
    async nodeCreated(node) {
        // Check if the node's comfyClass starts with "FL_"
        if (node.comfyClass.startsWith("FL_")) {
            // Apply styling
            node.color = "#16727c";
            node.bgcolor = "#4F0074";


        }
    }
});