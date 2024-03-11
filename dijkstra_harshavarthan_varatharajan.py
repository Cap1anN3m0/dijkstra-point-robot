import numpy as np
import heapq
import cv2
import math

#define the size of canvas
height = 500
width = 1200
explored_nodes = []#node to store the explored nodes
canvas = np.zeros((500, 1200, 3), dtype="uint8")#create empty canvas

#define the rectange dimensions and its clearance
rectangle = [
    {"clearance": ((100, 0), (175, 400)), "rectangle": ((105, 5), (170, 395))},
    {"clearance": ((275, 100), (350, 500)), "rectangle": ((280, 105), (345, 495))},
    {"clearance": ((895, 45), (1105, 130)), "rectangle": ((900, 50), (1100, 125))},
    {"clearance": ((1015, 120), (1105, 455)), "rectangle": ((1020, 125), (1100, 450))},
    {"clearance": ((895, 370), (1105, 455)), "rectangle": ((900, 375), (1100, 450))}
]

#function to draw the clearance
def draw_clearance(canvas, top_left, bottom_right, color=(0, 0, 0)):
    cv2.rectangle(canvas, top_left, bottom_right, color, -1)
#function to draw the rectangle
def draw_rectangle(canvas, top_left, bottom_right, color=(0, 255, 0)):
    cv2.rectangle(canvas, top_left, bottom_right, color, -1)
#function to calculate the vertices to draw the hexagon
def calculate_hex_vertices(center, size):
    vertices = []
    for i in range(6):
        angle_deg = 60 * i - 30#rotate the hexagon as given
        angle_rad = math.pi / 180 * angle_deg#convert to rad
        x = center[0] + size * math.cos(angle_rad)#find vector x
        y = center[1] + size * math.sin(angle_rad)#find vector y
        vertices.append((int(x), int(y)))
    return vertices

#define functions for movement
def move_up(node):
    return (node[0], node[1] - 1), 1.0
def move_down(node):
    return (node[0], node[1] + 1), 1.0
def move_left(node):
    return (node[0] - 1, node[1]), 1.0
def move_right(node):
    return (node[0] + 1, node[1]), 1.0
def move_up_left(node):
    return (node[0] - 1, node[1] - 1), 1.4
def move_up_right(node):
    return (node[0] + 1, node[1] - 1), 1.4
def move_down_left(node):
    return (node[0] - 1, node[1] + 1), 1.4
def move_down_right(node):
    return (node[0] + 1, node[1] + 1), 1.4

#define the action functions
action_functions = {
    'up': move_up,
    'down': move_down,
    'left': move_left,
    'right': move_right,
    'up_left': move_up_left,
    'up_right': move_up_right,
    'down_left': move_down_left,
    'down_right': move_down_right
}

#function to check if a node is free(not an obstacle or out of bounds)
def is_free(x, y, canvas):
    # Invert y-coordinate
    inverted_y = height - y - 1
    if 0 <= x < width and 0 <= inverted_y < height:
        # Check if the pixel is white (background)
        return np.array_equal(canvas[inverted_y, x], [255, 255, 255])
    return False

#dijkstra's algorithm to find the shortest path
def dijkstra(canvas, start, goal):
    queue = []#queue to add the nodes
    heapq.heappush(queue, (0, start))
    visited = set()#track visited nodes
    parents = {start: None}#dictionary to store the parent nodes
    distance = {start: 0}#dictionary to store the distance between nodes

    while queue:#continue until empty
        current_distance, current_node = heapq.heappop(queue)
        visited.add(current_node)#mark current as visited
        #check if current node is goal or not
        if current_node == goal:
            break
        #go through all nodes
        for action_name, action_function in action_functions.items():
            next_node, action_cost = action_function(current_node)#get the next node and the cost
            #invert y-coordinate for the check
            if is_free(next_node[0], height - next_node[1] - 1, canvas):
                next_distance = current_distance + action_cost
                #check if not visited or a shorter path to it is found, update the path and distance
                if next_node not in visited and (next_node not in distance or next_distance < distance[next_node]):
                    distance[next_node] = next_distance
                    parents[next_node] = current_node
                    explored_nodes.append(current_node)#add current node to the explored nodes
                    heapq.heappush(queue, (next_distance, next_node))

    path = []
    node = goal
    while parents[node] is not None:
        path.append(node)#add node to the path
        node = parents[node]#move to the parent node
    path.append(start)
    path.reverse()#reverse to start from beginning
    #return the path,cost and explored nodes
    return path, distance[goal] if goal in distance else None, explored_nodes


#backtracking to find the path
def backtrack(start, goal, parents):
    path = [goal]
    while path[-1] != start:
        path.append(parents[path[-1]])
    path.reverse()#reverse the order
    return path

#function to visualize the path
def visualize(canvas_BGR, path):
    #create the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('dijkstra_harshavarthan_varatharajan.mp4', fourcc, 20.0, (width, height))
    c1 = 0#counter for explored nodes
    c2 = 0#counter for node path
    cv2.circle(canvas_BGR, (start[0], height-start[1]-1), 5, (0, 255, 0), -1)
    #draw the goal in blue
    cv2.circle(canvas_BGR, (goal[0], height-goal[1]-1), 5, (255, 0, 0), -1)
    #draw explored nodes in yellow
    for node in explored_nodes:
            cv2.circle(canvas_BGR, (node[0],  height - node[1] - 1 ), 0, (0, 255, 255))
            c1 = c1+1
            if c1 % 2000 == 0:#append values in interval for fast visualization
                video.write(canvas_BGR)
    cv2.circle(canvas_BGR, (start[0], height - start[1] ), 5, (0, 255, 0), -1)
    #draw the goal in blue
    cv2.circle(canvas_BGR, (goal[0], height - goal[1] - 1), 5, (255, 0, 0), -1)
    #draw the path in red
    for node in path:
        cv2.circle(canvas_BGR, (node[0], height - node[1] - 1), 0, (0, 0, 255), )
        c2 = c2 + 1
        if c2 % 20 == 0:
            video.write(canvas_BGR)
    for i in range(35):
        video.write(canvas_BGR)
    #release the video writer
    video.release()

    #display the canvas
    cv2.imshow('Final output', canvas_BGR)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Main execution block
if __name__ == "__main__":
    #create the white background
    draw_clearance(canvas, (5, 5), (1195, 495), color=(255, 255, 255))
    #draw the clearance for the rectangles
    for rect in rectangle:
        draw_clearance(canvas, *rect["clearance"])
    #draw the solid rectangles
    for rect in rectangle:
        draw_rectangle(canvas, *rect["rectangle"])

    #draw hexagons
    outer_hex = calculate_hex_vertices((600, 250), 155.77)  # Slightly larger for clearance
    inner_hex = calculate_hex_vertices((600, 250), 150)

    #draw the outer hexagon in black for clearance
    cv2.fillPoly(canvas, [np.array(outer_hex)], color=(0, 0, 0))
    cv2.fillPoly(canvas, [np.array(inner_hex)], color=(0, 255, 0))
    #get the start and goal coordinates
    start = tuple(map(int, input("Enter start coordinates(x,y): ").split(',')))
    goal = tuple(map(int, input("Enter goal coordinates(x,y): ").split(',')))

    #validate the start and goal positions
    if not is_free(start[0], start[1], canvas) or not is_free(goal[0], goal[1], canvas):
        print("Enter valid coordinates, given start or goal is invalid")
    else:
        #flip the image 
        image = cv2.flip(canvas, 0)
        #initialize the pathplaninng algorithm and get the path, explored and cost to visualize
        path, cost, explored_nodes = dijkstra(image, start, goal)
        # canvas_BGR = cv2.cvtColor((canvas * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
        if path:
            print(f"Node path: {path}")#print the path
            print(f"cost: {cost}")#print the cost
            visualize(canvas, path)#initialize the visualization
        else:
            print("cant find the path")
