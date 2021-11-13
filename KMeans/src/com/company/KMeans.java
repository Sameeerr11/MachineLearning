package src.com.company;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class KMeans extends Pair {
    public static void main(String [] args){
        if (args.length < 3){
            System.out.println("Usage: Kmeans <input-image> <k> <output-image>");
            return;
        }
        try{
            BufferedImage originalImage = ImageIO.read(new File(args[0]));
            int k=Integer.parseInt(args[1]);
            BufferedImage kmeansJpg = kmeans_helper(originalImage,k);
            ImageIO.write(kmeansJpg, "jpg", new File(args[2]));

        }catch(IOException e){
            System.out.println(e.getMessage());
        }
    }

    private static BufferedImage kmeans_helper(BufferedImage originalImage, int k){
        int w=originalImage.getWidth();
        int h=originalImage.getHeight();
        BufferedImage kmeansImage = new BufferedImage(w,h,originalImage.getType());
        Graphics2D g = kmeansImage.createGraphics();
        g.drawImage(originalImage, 0, 0, w,h , null);
        // Read rgb values from the image
        int[] rgb=new int[w*h];
        int count=0;
        for(int i=0;i<w;i++){
            for(int j=0;j<h;j++){
                rgb[count++]=kmeansImage.getRGB(i,j);
            }
        }
        // Call kmeans algorithm: update the rgb values
        kmeans(rgb,k);

        // Write the new rgb values to the image
        count=0;
        for(int i=0;i<w;i++){
            for(int j=0;j<h;j++){
                kmeansImage.setRGB(i,j,rgb[count++]);
            }
        }
        return kmeansImage;
    }

    // Your k-means code goes here
    // Update the array rgb by assigning each entry in the rgb array to its cluster center
    private static void kmeans(int[] rgb, int k){
        Random rand = new Random();
        int[] centroidArray = new int[k];
        int[] centroidArray_Current = new int[k];
        int[] centroidArray_New = new int[k];

        List<Pair> rgbList = new ArrayList<>();

        //Making a list of pairs
        for(int i=0; i< rgb.length; i++){
            Pair pair = new Pair();
            pair.setKey(rgb[i]);
            rgbList.add(pair);
        }

        //Initialize random centroids
        for(int i=0; i<k; i++) {
            int random = rand.nextInt(rgb.length-1);
            centroidArray[i] = rgbList.get(random).getKey();
        }

        int iter=0;                                             //to check when it converges//not needed for code to run
        while(true) {                                           //iterate until convergence
            boolean convergence = true;

            for(int i=0; i<centroidArray.length;i++){
                centroidArray_Current[i] = centroidArray[i];
            }

            //Assigning the centroid value which is closest to the key value
            for (int i = 0; i < rgbList.size(); i++) {
                int minValue = 0;
                int rgbValue = rgbList.get(i).getKey();
                int rgbBlue= rgbValue & 0xff;
                int rgbGreen = (rgbValue & 0xff00) >> 8;
                int rgbRed = (rgbValue & 0xff0000) >> 16;

                int minDistance = Integer.MAX_VALUE;
                for (int j = 0; j < centroidArray.length; j++) {
                    int centroidValue = centroidArray[j];
                    int centroid_B= centroidValue & 0xff;
                    int centroid_G = (centroidValue & 0xff00) >> 8;
                    int centroid_R = (centroidValue & 0xff0000) >> 16;

                    int pointsDistance = (int) Math.sqrt(Math.pow(Math.abs(centroid_B-rgbBlue),2)+Math.pow(Math.abs(centroid_G-rgbGreen),2)+Math.pow(Math.abs(centroid_R-rgbRed),2));
                    if (pointsDistance < minDistance) {
                        minDistance = pointsDistance;
                        minValue = centroidValue;
                    }
                }
                rgbList.get(i).setValue(minValue); //sets minimum centroid value after iterating over all centroid values
            }

            //Find common centroids, take their mean and make them as new centroids
            for (int i = 0; i < centroidArray.length; i++) {
                int x = centroidArray[i];
                long b1 = 0, r1 = 0, g1 = 0;
                int count = 0;
                for (int j = 0; j < rgbList.size(); j++) {
                    if (rgbList.get(j).getValue() == x) {
                        int rgbPixelKey = rgbList.get(j).getKey();
                        int pixel_B= rgbPixelKey & 0xff;
                        int pixel_G = (rgbPixelKey & 0xff00) >> 8;
                        int pixel_R = (rgbPixelKey & 0xff0000) >> 16;
                        b1 += pixel_B;
                        g1 += pixel_G;
                        r1 += pixel_R;
                        count++;
                    }
                }
                //Calculate Centroid Mean value and save it as new Centroid
                int meanCentroidRed = (int) (r1 / count);
                int meanCentroidGreen = (int) (g1 / count);
                int meanCentroidBlue = (int) (b1 / count);

                //Calculate independent means and then convert them to a single int rgb value
                int newCentroidValue = new Color(meanCentroidRed, meanCentroidGreen, meanCentroidBlue).getRGB();
                centroidArray[i] = newCentroidValue;
                centroidArray_New[i] = newCentroidValue;
            }
            for(int i=0; i<centroidArray.length;i++){           //check for convergence
                if(centroidArray_New[i] != centroidArray_Current[i]){
                    convergence = false;
                }
            }
            if(convergence) {                       //Break if converged
                System.out.println("Converged...at "+iter);
                break;
            }
            iter++;
        }
        //Updating the RGB values to the array from the RGBList
        for(int i=0; i<rgb.length; i++){
            rgb[i] = rgbList.get(i).getValue();
        }

    }


}
