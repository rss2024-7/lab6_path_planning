import cv2
import sys
import numpy as np

convex_sets = [
    np.array([[0, 280],
              [0, 345],
              [95, 345],
              [45, 280]]),
    np.array([[50, 286],
              [101, 286],
              [101, 345],
              [50, 345]]),
    np.array([[91, 280],
              [83, 286],
              [100, 345],
              [148, 345],
              [95, 280]]),
    np.array([[106, 294],
              [250, 345],
              [106, 345]]),
    np.array([[125, 300],
              [175, 300],
              [175, 345]]),
    np.array([[174, 300],
              [191, 280],
              [270, 280],
              [325, 345],
              [174, 345]]),
    np.array([[284, 303],
              [491, 303],
              [491, 345],
              [284, 345]]),
    np.array([[333, 344],
              [341, 361],
              [347, 361],
              [355, 344]]),
    np.array([[495, 287],
              [455, 345],
              [540, 345]]),
    np.array([[488, 280],
              [529, 280],
              [565, 345],
              [527, 345]]),
    np.array([[533, 291],
              [552, 290],
              [552, 345],
              [533, 345]]),
    np.array([[533, 291],
              [833, 290],
              [867, 348],
              [533, 344]]),
    np.array([[824, 282],
              [922, 432],
              [983, 286]]),
    np.array([[950, 286],
              [1121, 286],
              [1121, 352],
              [950, 352]]),
    np.array([[1118, 305],
              [1349, 305],
              [1349, 356],
              [1118, 356]]),
    np.array([[1192, 283],
              [1207, 283],
              [1220, 306],
              [1173, 318]]),
    np.array([[1306, 288],
              [1684, 295],
              [1676, 344],
              [1333, 343]]),
    np.array([[1685, 296],
              [1698, 306],
              [1566, 474],
              [1569, 294]]),
    np.array([[1633, 384],
              [1633, 461],
              [1579, 480],
              [1566, 471]]),
    np.array([[1586, 476],
              [1618, 464],
              [1618, 509],
              [1586, 606]]),  
    np.array([[1632, 509],
              [1632, 536],
              [1586, 606],
              [1610, 509]]),
    np.array([[1621, 546],
              [1631, 559],
              [1630, 1061],
              [1566, 1061],
              [1563, 629]]),  
    np.array([[1698, 1039],
              [1621, 1039],
              [1621, 1010],
              [1698, 1010]]), 
    np.array([[1651, 996],
              [1642, 1013],
              [1698, 1013],
              [1698, 996]]),   
    np.array([[1628, 680],
              [1628, 689],
              [1663, 689],
              [1663, 680]]),
    np.array([[1571, 1000],
              [1571, 1018],
              [1517, 1018],
              [1517, 1000]]),
    np.array([[903, 996],
              [903, 1027],
              [1521, 1024],
              [1521, 996]]),
    np.array([[903, 1027],
              [899, 897],
              [920, 870],
              [927, 879],
              [932, 1000]]),
    np.array([[1446, 961],
              [1457, 965],
              [1440, 1000],
              [1419, 1004]]),
    np.array([[1443, 983],
              [1461, 1000],
              [1436, 1003]]),
    np.array([[905, 910],
              [905, 835],
              [926, 835],
              [926, 847]]),
    np.array([[913, 835],
              [913, 857],
              [530, 857],
              [530, 835]]),
    np.array([[761, 868],
              [742, 852],
              [777, 854]]),
    np.array([[532, 876],
              [532, 854],
              [570, 854],
              [562, 876]]),
    np.array([[533, 838],
              [538, 787],
              [651, 787],
              [660, 838]]),
    np.array([[583, 792],
              [653, 792],
              [631, 742]]),
    np.array([[869, 341],
              [961, 341],
              [883, 528],
              [868, 487]]),
    np.array([[873, 478],
              [884, 519],
              [866, 534],
              [842, 514]]),
    np.array([[886, 514],
              [921, 530],
              [911, 537],
              [878, 526]]),
    np.array([[844, 511],
              [814, 543],
              [823, 546],
              [854, 516]]),
    np.array([[767, 549],
              [820, 549],
              [767, 599]]),
    np.array([[822, 534],
              [846, 556],
              [671, 747],
              [642, 726]]),
    np.array([[635, 714],
              [635, 774],
              [676, 732]]),
    np.array([[745, 687],
              [760, 670],
              [749, 657],
              [731, 675]]),
    np.array([[697, 632],
              [681, 650],
              [695, 677],
              [729, 641]]),
    np.array([[639, 692],
              [665, 706],
              [719, 660]]),
    np.array([[630, 747],
              [657, 696],
              [678, 708],
              [632, 759]]),
]

def convert_polygons_to_map_coordinates(pixel_polygons):
    # (0,0) in pixel coordinates maps to (25.795 -17.009) in map coordinates
    # (514, 335) in map coordinates maps to (0,0) in pixel coordinates

    # Known mapping points
    pixel_origin = np.array([0, 0])
    map_origin = np.array([25.795, -17.009])
    pixel_max = np.array([514, 335])
    map_max = np.array([0, 0])

    # Calculate scale factors
    scale_x = (map_max[0] - map_origin[0]) / (pixel_max[0] - pixel_origin[0])
    scale_y = (map_max[1] - map_origin[1]) / (pixel_max[1] - pixel_origin[1])

    # Calculate the translation components
    translate_x = map_origin[0] - scale_x * pixel_origin[0]
    translate_y = map_origin[1] - scale_y * pixel_origin[1]

    map_polygons = []
    for poly in pixel_polygons:
        # Apply the transformation to each point in the polygon
        transformed = (poly * np.array([scale_x, scale_y])) + np.array([translate_x, translate_y])
        map_polygons.append(transformed)

    return map_polygons

convex_sets = convert_polygons_to_map_coordinates(convex_sets)



if __name__ == "__main__":

    def pixel_to_real(pixel):
        # need to check if this returns [u, v] or [v, u]
        x = 25.900000
        y = 48.50000
        theta = 3.14
        scale = 0.0504
        transform = np.array([[np.cos(theta), -np.sin(theta), x],
                    [np.sin(theta), np.cos(theta), y],
                    [0,0,1]])

        pixel = np.array([pixel[0], pixel[1]]) * scale
        pixel = np.array([*pixel,1])
        pixel = np.linalg.inv(transform) @ pixel
        point = pixel
        # return point
        return point[:2]
    
    x = np.array([[488, 280],
              [529, 280],
              [565, 345],
              [527, 345]])
    
    print(np.apply_along_axis(pixel_to_real, axis=0, arr=x.T))

    def draw_translucent_polygon(img, points, color, opacity=0.5):
        """
        Draw a polygon with translucent filling on an image.

        Parameters:
        - img: Source image.
        - points: A d x 2 matrix of points (numpy array) representing the vertices of the polygon.
        - color: A tuple representing the color of the polygon (B, G, R).
        - opacity: Opacity of the polygon fill, where 0 is fully transparent and 1 is fully opaque.

        Returns:
        - An image with the translucent polygon.
        """
        # Create an overlay image the same size as the original
        overlay = img.copy()
        
        # Reshape points to a format required by polylines and fillPoly
        pts = points.reshape((-1, 1, 2))
        
        # Draw the polygon on the overlay
        cv2.fillPoly(overlay, [pts], color)
        
        # Blend the overlay with the original image using the opacity
        cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
        
        return img
        
    image = cv2.imread("map.png")

    for polytope in convex_sets:
        image = draw_translucent_polygon(image, polytope, color=tuple(int(c) for c in np.random.choice(range(256), size=3)))


    cv2.namedWindow('Fullscreen Window', cv2.WINDOW_NORMAL)

    # cv2.setWindowProperty('Fullscreen Window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Fullscreen Window', image)

    # Loop until the specific key is pressed
    while True:
        # Wait for the 'q' key to be pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()