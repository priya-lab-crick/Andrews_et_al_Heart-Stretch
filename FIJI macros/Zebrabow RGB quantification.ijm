rename("Image");

run("Split Channels");

selectWindow("C1-Image");

	for (i=0 ; i<roiManager("count"); i++) {
    			        roiManager("select", i);
    			        roiManager("multi-measure append");
	}
	
	selectWindow("C2-Image");

	for (i=0 ; i<roiManager("count"); i++) {
    			        roiManager("select", i);
    			        roiManager("multi-measure append");
	}
	
	selectWindow("C4-Image");

	for (i=0 ; i<roiManager("count"); i++) {
    			        roiManager("select", i);
    			        roiManager("multi-measure append");
	}
	
