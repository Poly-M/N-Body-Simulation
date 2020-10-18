#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <time.h> 
#include <mpi.h> 
#include <iostream>
#include <fstream>
using namespace std;

#define DEFAULT_N 10000   // Number of particles
#define DEFAULT_TIME 1000 // Number if iterations
#define G 6.67300e-11     // Gravitational constant, m3 kg-1 s-2
#define XBOUND 1.0e6      // Width of space
#define YBOUND 1.0e6      // Height of space
#define ZBOUND 1.0e6      // Depth of space
#define RBOUND 10         // Upper bound on radius
#define DELTAT 0.01       // Time increament
#define THETA 1.0         // Opening angle, for approximation in BH
#define MASS_OF_UNKNOWN 1.899e12   //  for m.


//Position of Particle
typedef struct {
   double px, py, pz;
} Position;


//Velocity of Particle
typedef struct {
   double vx, vy, vz;
} Velocity;


//Force of Particle
typedef struct {
   double fx, fy, fz;
} Force;


// Cell representing a node for Barnes Hut
typedef struct Cell  {

   int index;                    // Index into arrays to identify particle's 
                                 // position and mass
   int no_subcells;              // Indicate whether cell is leaf or not
   double mass;                  // Mass of particle of total mass of subtree
   double x, y, z;               // Location of cell(cube) in space
   double cx, cy, cz;            // Location of center of mass of cell
   double width, height, depth;  // Width, Height, and Depth of cell
   struct Cell* subcells[8];     // Pointers to child nodes

} Cell;


Position* position;   // Current positions for all particles
Velocity* ivelocity;  // Initial velocity for all particles
Velocity* velocity;   // Velocity of particles in current processor
double* mass;         // Mass of each particle
double* radius;       // Radius of each particle
Force* force;         // Force experienced by all particles
Cell* root_cell;      // Root of BH octtree


// MPI Datatype to exchange particle data
MPI_Datatype MPI_POSITION;
MPI_Datatype MPI_VELOCITY;


int N;               // User specified particle count
int TIME;            // User specified iterations
int ranks;            // Rank of process
int size;            // Number of processes in the group
int part_size;       // Number of particles each processor is responsible for
int pindex;          // The pindex points to the slot in the vectors/arrays that contains 
                     // data concerning the current processor, i.e pindex = (rank * part_size)


int name_length;                    // Length of processor name
char name[MPI_MAX_PROCESSOR_NAME];  // Buffer to hold processor name


// Random number between 0 to 1
double generate_rand(){
   return rand()/((double)RAND_MAX + 1);
}


// Random number between -1 to 1
double generate_rand_ex(){
   return 2 * generate_rand() - 1;
}


// Check if two particless collide
int check_collision(int index1, int index2) {

   if (pow((position[index1].px - position[index2].px), 2.0) + 
       pow((position[index1].py - position[index2].py), 2.0) +
       pow((position[index1].pz - position[index2].py), 2.0) <
       pow((radius[index1] + radius[index2]), 2.0)) {
       
       // Collision detected
       return 1;

   }   
   return 0;
}


// Computes distance between two particles
double compute_distance(Position a, Position b){
    return sqrt(pow((a.px - b.px), 2.0) +
               pow((a.py - b.py), 2.0) +
               pow((a.pz - b.pz), 2.0));
}


// Initialize space with random positions, mass, and velocity
void initialize_space() {   
   
   // Inner bounds to prevent generating a particle whose
   // surface lies outside the boundaries of space
   double ixbound = XBOUND - RBOUND;
   double iybound = YBOUND - RBOUND;
   double izbound = ZBOUND - RBOUND;


   for (int i = 0; i < N; i++) {
      mass[i] = MASS_OF_UNKNOWN * generate_rand();
      radius[i] = RBOUND * generate_rand();
      position[i].px = generate_rand() * ixbound;
      position[i].py = generate_rand() * iybound;
      position[i].pz = generate_rand() * izbound;
      ivelocity[i].vx = generate_rand_ex();
      ivelocity[i].vy = generate_rand_ex();
      ivelocity[i].vz = generate_rand_ex();; 
   }
   
   // Check for particles that had already collided
   for (int i = 0; i < N; i++) {

      for (int j = i + 1; j < N; j++) {

         if (check_collision(i, j)) {
            double d = compute_distance(position[i], position[j]);
            radius[i] = radius[j] = d/2.0;
         }
      }
   }
}


// Compute gravitational forces each particles goes through by other bodies
void compute_force(){

   for (int i = 0; i < part_size; i++) {

      force[i].fx = 0.0;
      force[i].fy = 0.0;
      force[i].fz = 0.0;

      for (int j = 0; j < N; j++){

         if (j == (i + pindex)) continue; // avoid computation for 
                                          // same bodies

         double d = compute_distance(position[i + pindex], position[j]);

         // Compute grativational force according to Newtonian's law
         double f = (G * (mass[i + pindex] * mass[j]) / 
                         (pow(d, 2.0)));

         // Resolve forces in each direction
         force[i].fx += f * ((position[j].px - position[i + pindex].px) / d);
         force[i].fy += f * ((position[j].py - position[i + pindex].py) / d);
         force[i].fz += f * ((position[j].pz - position[i + pindex].pz) / d);
      }
   }
}


// Compute the new velocities due to forces
void compute_velocity(){
   for (int i = 0; i < part_size; i++) {
      velocity[i].vx += (force[i].fx / mass[i + pindex]) * DELTAT;
      velocity[i].vy += (force[i].fy / mass[i + pindex]) * DELTAT;
      velocity[i].vz += (force[i].fz / mass[i + pindex]) * DELTAT;
   }
}


// Compute the new positions in space
void compute_positions(){
   for (int i = 0; i < part_size; i++) {
      position[i + pindex].px += velocity[i].vx * DELTAT;
      position[i + pindex].py += velocity[i].vy * DELTAT;
      position[i + pindex].pz += velocity[i].vz * DELTAT;

      // Check if particles attempt to cross boundary      
      if ((position[i + pindex].px + radius[i + pindex]) >= XBOUND ||
          (position[i + pindex].px - radius[i + pindex]) <= 0)
         velocity[i].vx *= -1;
      else if ((position[i + pindex].py + radius[i + pindex] >= YBOUND) || 
               (position[i + pindex].py - radius[i + pindex]) <= 0)
         velocity[i].vy *= -1;
      else if ((position[i + pindex].pz + radius[i + pindex]) >= ZBOUND || 
               (position[i + pindex].pz - radius[i + pindex]) <= 0)
         velocity[i].vz *= -1;      
   }
}


// Create a cell for use in octree
Cell* create_cell(double width, double height, double depth) {

   Cell* cell = (Cell*) malloc(sizeof(Cell));
   cell->mass = 0;
   cell->no_subcells = 0;
   cell->index = -1;
   cell->cx = 0;
   cell->cy = 0;
   cell->cz = 0;
   cell->width = width;
   cell->height = height;
   cell->depth = depth;   
   return cell;
}


// Set the location of the subcell wrt the current cell
void BH_set_location_of_subcells(Cell* cell, double width, double heigth, double depth){

   // Set location of new cells
   cell->subcells[0]->x = cell->x;
   cell->subcells[0]->y = cell->y;
   cell->subcells[0]->z = cell->z;

   cell->subcells[1]->x = cell->x + width;
   cell->subcells[1]->y = cell->y;
   cell->subcells[1]->z = cell->z;

   cell->subcells[2]->x = cell->x + width;
   cell->subcells[2]->y = cell->y;
   cell->subcells[2]->z = cell->z + depth;

   cell->subcells[3]->x = cell->x;
   cell->subcells[3]->y = cell->y;
   cell->subcells[3]->z = cell->z + depth;

   cell->subcells[4]->x = cell->x;
   cell->subcells[4]->y = cell->y + heigth;
   cell->subcells[4]->z = cell->z;

   cell->subcells[5]->x = cell->x + width;
   cell->subcells[5]->y = cell->y + heigth;
   cell->subcells[5]->z = cell->z;

   cell->subcells[6]->x = cell->x + width;   // Coordinates of this cell marks
   cell->subcells[6]->y = cell->y + heigth;  // the mid-point of the parent cell
   cell->subcells[6]->z = cell->z + depth;   
   
   cell->subcells[7]->x = cell->x;
   cell->subcells[7]->y = cell->y + heigth;
   cell->subcells[7]->z = cell->z + depth;
}


// Generates subcell for current cell
void BH_generate_subcells(Cell* cell) {
   
   // Calculate subcell dimensions
   double width  = cell->width / 2.0;
   double height = cell->height / 2.0;
   double depth  = cell->depth / 2.0;

   // Cell no longer a leaf
   cell->no_subcells = 8;   
   
   // Create and initialize new subcells   
   for (int i = 0; i < cell->no_subcells; i++) {
      cell->subcells[i] = create_cell(width, height, depth);
   }
   
   BH_set_location_of_subcells(cell, width, height, depth);   
}


// Locate subcell to which particle must be added
int BH_locate_subcell(Cell* cell, int index) {

   // Determine which subcell to add the body to
   if (position[index].px > cell->subcells[6]->x){
      if (position[index].py > cell->subcells[6]->y){
         if (position[index].pz > cell->subcells[6]->z)
            return 6;
         else
            return 5;
      }
      else{
         if (position[index].pz > cell->subcells[6]->z)
            return 2;
         else
            return 1;
      }
   }
   else{
      if (position[index].py > cell->subcells[6]->y){
         if (position[index].pz > cell->subcells[6]->z)
            return 7;
         else
            return 4;
      }
      else{
         if (position[index].pz > cell->subcells[6]->z)
            return 3;
         else
            return 0;
      }      
   }
}


// Add a particle to the cell
// If a particle already exists, subdivide
void BH_add_to_cell(Cell* cell, int index) {

   if (cell->index == -1) {         
      cell->index = index;
      return;         
   }
         
   BH_generate_subcells(cell);

   // The current cell's body must now be re-added to one of its subcells
   int sc1 = BH_locate_subcell(cell, cell->index);
   cell->subcells[sc1]->index = cell->index;   

   // Locate subcell for new body
   int sc2 = BH_locate_subcell(cell, index);

   if (sc1 == sc2)
      BH_add_to_cell(cell->subcells[sc1], index);
   else 
      cell->subcells[sc2]->index = index;  
}

// Generate octree for entire system
void BH_generate_octtree() {
   
   // Initialize root of octtree
   root_cell = create_cell(XBOUND, YBOUND, ZBOUND);
   root_cell->index = 0;
   root_cell->x = 0;
   root_cell->y = 0;
   root_cell->z = 0;
   
   int i;
   for (i = 1; i < N; i++) {

      Cell* cell = root_cell;

      // Find which node to add the body to
      while (cell->no_subcells != 0){
         int sc = BH_locate_subcell(cell, i);
         cell = cell->subcells[sc];
      }      

      BH_add_to_cell(cell, i);
   }
}


// Compute total mass and center of mass
Cell* BH_compute_cell_properties(Cell* cell){
   
   if (cell->no_subcells == 0) {
      if (cell->index != -1){
         cell->mass = mass[cell->index];
         return cell;
      }
   }
   else {      
      double tx = 0, ty = 0, tz = 0;
      for (int i = 0; i < cell->no_subcells; i++) {
         Cell* temp = BH_compute_cell_properties(cell->subcells[i]);
         if (temp != NULL) {
            cell->mass += temp->mass;
            tx += position[temp->index].px * temp->mass;
            ty += position[temp->index].py * temp->mass;
            tz += position[temp->index].pz * temp->mass;            
         }
      }
      
      // Compute center of mass
      cell->cx = tx / cell->mass;
      cell->cy = ty / cell->mass;
      cell->cz = tz / cell->mass;
   
      return cell;
   }
   return NULL;
}


// Compute force experienced between cell and particle
void BH_compute_force_from_cell(Cell* cell, int index) {
   double d = compute_distance(position[index], position[cell->index]);

   // Compute grativational force according to Newtonian's law
   double f = (G * (mass[index] * mass[cell->index]) / 
                   (pow(d, 2.0)));

   // Resolve forces in each direction
   force[index - pindex].fx += f * ((position[cell->index].px - position[index].px) / d);
   force[index - pindex].fy += f * ((position[cell->index].py - position[index].py) / d);
   force[index - pindex].fz += f * ((position[cell->index].pz - position[index].pz) / d);      
}


// Compute forces between particles using approximation
void BH_compute_force_from_octtree(Cell* cell, int index) {
   
   if (cell->no_subcells == 0) {
      if (cell->index != -1 && cell->index != index) {
         BH_compute_force_from_cell(cell, index);
      }
   }
   else {
      double d = compute_distance(position[index], position[cell->index]);
      
      if (THETA > (cell->width / d)){ 
         // Use approximation
         BH_compute_force_from_cell(cell, index);         
      }
      else {
         for (int i = 0; i < cell->no_subcells; i++) {
            BH_compute_force_from_octtree(cell->subcells[i], index);
         }
      }      
   }
}


// Compute using Barnes Hut
void BH_compute_force(){

   for (int i = 0; i < part_size; i++) {

      force[i].fx = 0.0;
      force[i].fy = 0.0;
      force[i].fz = 0.0;

      BH_compute_force_from_octtree(root_cell, i + pindex);
   }
}


// Deletes the octtree
void BH_delete_octtree(Cell* cell) {
   
   if (cell->no_subcells == 0) {
      free(cell);
      return;
   }

   int i;
   for (i = 0; i < cell->no_subcells; i++) {
      BH_delete_octtree(cell->subcells[i]);
   }

   free(cell);
}


void write_positions() {

   // Create a file and output
   std::ofstream file;
<<<<<<< HEAD
   file.open("positions.dat");
=======
   file.open("positions.dat", "w");
>>>>>>> df8504a9d9197f70c2363016115eac1a04c12e32
   
   int i;
   for (i = 0; i < N; i++) {
       file << "px=" << position[i].px << ", py=" << position[i].py << ", pz=" << position[i].pz << endl;
   }
}


void init_velocity(){
   for (int i = 0; i < part_size; i++){
      velocity[i].vx = 0;
      velocity[i].vy = 0;
      velocity[i].vz = 0;
   }
}


void run_simulation(){

   if (ranks == 0)
      std::cout << ("\nRunning simulation for %d bodies with %d iterations, and DELTAT = %f..\n\n", N, TIME, DELTAT);
   

   // Broadcast mass and position to all members in the group
   MPI_Bcast(mass, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   MPI_Bcast(position, N, MPI_POSITION, 0, MPI_COMM_WORLD);
   MPI_Scatter(ivelocity, part_size, MPI_VELOCITY, velocity, part_size, MPI_VELOCITY, 0, MPI_COMM_WORLD);


   int i;

   for (i = 0; i < TIME; i++) {
      BH_generate_octtree();
      BH_compute_cell_properties(root_cell);
      BH_compute_force();
      BH_delete_octtree(root_cell);

      compute_velocity();
      compute_positions();
      MPI_Allgather(position + (ranks * part_size), part_size, MPI_POSITION, 
                    position, part_size, MPI_POSITION, MPI_COMM_WORLD); 
   }

   if (ranks == 0)
      write_positions();

}


int main(int argc, char* argv[]){

   // Initialize MPI execution env.
   MPI_Init(&argc, &argv);

   double start = MPI_Wtime();
   
   // Initialise problem parameters
   if (argc >= 2) 
      sscanf(argv[1], "%i%", &N);
   else
      N = DEFAULT_N;

   if (argc >= 3)
      sscanf(argv[2], "%i%", &TIME);   
   else
      TIME = DEFAULT_TIME;


   // Get rank and size
   MPI_Comm_rank(MPI_COMM_WORLD, &ranks);
   MPI_Comm_size(MPI_COMM_WORLD, &size);


   // Create and commit new MPI Types
   MPI_Type_contiguous(3, MPI_DOUBLE, &MPI_POSITION);
   MPI_Type_contiguous(3, MPI_DOUBLE, &MPI_VELOCITY);
   MPI_Type_commit(&MPI_POSITION);
   MPI_Type_commit(&MPI_VELOCITY);

   // Number of bodies each processor is responsible for
   part_size = N / size;


   // Determine index into array structures for each process
   pindex = ranks * part_size;


   // Allocate memory for mass, disance, velocity and force arrays
   mass = (double *) malloc(N * sizeof(double));
   radius = (double *) malloc(N * sizeof(double));
   position = (Position *) malloc(N * sizeof(Position));
   ivelocity = (Velocity *) malloc(N * sizeof(Velocity));
   velocity = (Velocity *) malloc(part_size * sizeof(Velocity));
   force = (Force *) malloc(part_size * sizeof(Force));


   // Initialize velocity array for each process
   init_velocity();


   // Let the master initialize the space
   if (ranks == 0){
      initialize_space();
   }
   
   
   // Run the N-body simulation
   run_simulation();

   double end = MPI_Wtime();

   cout << "The process took " << end - start << " seconds to run." << endl;


   // Terminate MPI execution env.
   MPI_Finalize();

   return 0; 

}
