// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Andrei Orehov / Andromediastudio - https://andromediastudio.com
// comfyui-vae-float32 - https://github.com/AndreiOrehov/comfyui-vae-float32
//
// One house colour for every node in this pack, so it is visible at a glance which nodes in a graph
// are ours. There is no Python-side way to do this: server.py's node_info() sends the frontend a fixed
// set of fields (input/output/category/description/tooltips/deprecated/...) and colour is not among
// them. The frontend colours a node by assigning this.color / this.bgcolor at creation time - exactly
// how it paints API nodes yellow:
//     ComfyNode.nodeData.api_node && (this.color = node_colors.yellow.color, ...)
// so we do the same thing from here, for our own nodes only.
//
// Deliberately NOT done by setting API_NODE = True on the Python class to inherit the yellow: that flag
// also files the node under Partner Nodes and gives it a credits-per-run badge, which would be a lie.
import { app } from "../../scripts/app.js";

// Matched on CATEGORY rather than a list of class names, so a node added later is coloured for free.
const OUR_CATEGORY = "ANDRO";

// color = title bar, bgcolor = body. Both are darker than LiteGraph's defaults (#333 / #353535) on
// purpose: a node has to read as ours without shouting over the graph.
const HOUSE = { color: "#2a2340", bgcolor: "#3b3260" };

// 1.3.0 renamed every node key. The old keys stay registered on the Python side so nothing ever fails
// to load, but a graph saved before the rename would otherwise sit there wearing "(old name)" and a
// DEPRECATED badge on every node - which reads as breakage rather than as a rename that went fine.
// Rewriting the type as the graph is configured moves those nodes onto the current classes silently;
// the user only has to save the workflow once for it to be permanent.
const RENAMED = {
    VAEDecodeFloat32: "ANDROVAEDecode",
    VAEEncodeFloat32: "ANDROVAEEncode",
    ImageRangeStats: "ANDRORangeStats",
    ImageCompareNumeric: "ANDROCompare",
    TileSeamCheck: "ANDROSeamCheck",
    RemapRange: "ANDRORemapRange",
    SaveEXRFloat32: "ANDROSaveEXR",
    LoadAudioOptional: "ANDROLoadAudio",
    AudioLatentSwitch: "ANDROAudioSwitch",
};

// Nodes live in graphData.nodes, and also inside every subgraph definition - this pack's own example
// workflow keeps its decode chain in one, so missing that branch would leave exactly the graphs that
// matter untouched.
function migrate(container, counter) {
    for (const node of container?.nodes || []) {
        const to = RENAMED[node.type];
        if (to) {
            node.type = to;
            if (node.properties?.["Node name for S&R"]) node.properties["Node name for S&R"] = to;
            counter.n++;
        }
    }
    for (const sub of container?.definitions?.subgraphs || []) migrate(sub, counter);
}

app.registerExtension({
    name: "Andromedia.houseColour",
    beforeConfigureGraph(graphData) {
        const counter = { n: 0 };
        migrate(graphData, counter);
        if (counter.n) {
            console.log(`[ANDRO] migrated ${counter.n} node(s) from pre-1.3.0 names. ` +
                        `Save the workflow to keep it.`);
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.category !== OUR_CATEGORY) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            // Only newly created nodes. A node loaded from a saved workflow carries its own stored
            // colour and is left alone by design - the user's choice outranks the house style.
            this.color = HOUSE.color;
            this.bgcolor = HOUSE.bgcolor;
            return r;
        };
    },
});
