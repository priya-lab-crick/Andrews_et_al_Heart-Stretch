#@File(label = "Define data directory", style = "directory") Directory1
#@File(label = "Define save directory", style = "directory") Directory2

run("Close All");

fileNames = getFileList(Directory1);

for (i = 0; i < fileNames.length; i = i + 1) {

open(Directory1 + File.separator + fileNames[i]);
imageName = getTitle();

selectWindow(imageName);

rename("Image");

Stack.setXUnit("microns");
run("Properties...", "channels=2 slices=1 frames=1 pixel_width=0.2071607 pixel_height=0.2071607 voxel_depth=1");

run("Split Channels");

selectImage("C1-Image");
rename("Labels");

run("Analyze Regions", "area");

saveAs("Results", Directory2 + File.separator + fileNames[i]+"_Areas" + ".csv");

close("Results");

run("Label image to ROIs");
RoiManager.scale(0.8, 0.8, true);
roiManager("Combine");
run("Create Mask");

imageCalculator("Subtract create", "Labels", "Mask");

rename("Cortex");

run("glasbey inverted");

selectImage("Mask");
run("Invert");

imageCalculator("Subtract create", "Labels", "Mask");

rename("Cytoplasm");

run("glasbey inverted");

run("Intensity Measurements 2D/3D", "input=C2-Image labels=Cortex mean mode");

saveAs("Results", Directory2 + File.separator + fileNames[i]+"_Cortex.csv");

close("Results");

run("Intensity Measurements 2D/3D", "input=C2-Image labels=Cytoplasm mean mode");

saveAs("Results", Directory2 + File.separator + fileNames[i]+"_Cyto.csv");

close("Results");

selectWindow("Cortex");

run("RGB Color");

saveAs("Tiff", Directory2 + File.separator + fileNames[i]+"_CortexLabels");

selectWindow("Cytoplasm");

run("RGB Color");

saveAs("Tiff", Directory2 + File.separator + fileNames[i]+"_CytoLabels");

run("Close All");}




