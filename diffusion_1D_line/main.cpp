#include <iostream>

#include "util/PathsAndFiles.hpp"

#include "level_set/redistancing_Sussman/AnalyticalSDF.hpp" // Analytical SDF to define the disk-shaped diffusion domain
#include "level_set/redistancing_Sussman/HelpFunctionsForGrid.hpp"
#include "HelpFunctions_diffusion.hpp"

// #include "include/FD_laplacian.hpp"
// #include "../include/timesteps_stability.hpp"
// #include "../include/monitor_total_mass.hpp"


// Grid dimensions
const size_t dims = 1;

// Space indices
constexpr size_t x = 0, y = 1;

// Property indices
#if 0
constexpr size_t
PHI_SDF		= 0,

U1_N		= 1,
U1_NPLUS1	= 2,
LAP_U		= 3,
D1		= 4,

C2_N		= 5,
C2_NPLUS1	= 6,
C2_LAP		= 7,
D2		= 8,

IS_SOURCE	= 9,
K_SINK          = 10,

K_ON		= 11,
K_OFF		= 12,

K_DECAY		= 13;
#endif

constexpr size_t
PHI_SDF		= 0,
U1_N		= 1,
U1_NPLUS1	= 2,
LAP_U		= 3,
IS_SOURCE	= 4;


typedef aggregate<double, double, double, double, int> props;

// template<class T, std::size_t n>
// std::size_t len(T (&v)[n]) { return n; }

template<size_t FIELD, size_t DERIVATIVE, size_t STENCIL_SIZE, typename GRID_TYPE, typename KEYTYPE, typename FIELD_TYPE>
void finite_differences(const GRID_TYPE & grid,
						const KEYTYPE & key, 
						const int (& stencil) [STENCIL_SIZE], 
						const FIELD_TYPE (& coefficients) [STENCIL_SIZE], 
						const FIELD_TYPE & divisor,
						const size_t dimension)
{
	grid.template getProp<DERIVATIVE>(key) = 0;
	for(size_t neighbor_index = 0; neighbor_index < STENCIL_SIZE; ++neighbor_index)
	{
		grid.template getProp<DERIVATIVE>(key) += coefficients[neighbor_index] * grid.template get<FIELD>(key.move(dimension, stencil[neighbor_index]));
		std::cout << "coefficients[neighbor_index] = " << coefficients[neighbor_index] << std::endl;
		std::cout << "grid.template get<FIELD>(key.move(dimension, stencil[neighbor_index])) = " << grid.template get<FIELD>(key.move(dimension, stencil[neighbor_index])) << std::endl;
		std::cout << "stencil[neighbor_index] = " << stencil[neighbor_index] << std::endl;
		std::cout << "---------------------------------------------------------------" << std::endl;
	}
	grid.template getProp<DERIVATIVE>(key) /= divisor;
}

template<size_t FIELD, size_t LAP_U, size_t PHI_SDF, typename GRID_TYPE, typename BOUNDARY_TYPE>
void finite_differences_2nd_derivative_no_flux_BCs(const GRID_TYPE & grid,
												   const size_t dimension,
												   const BOUNDARY_TYPE b_low)
{
	const size_t stencil_size = 3;
	const int stencil [stencil_size] = {-1, 0, 1};
	// u''_i = (u_i-1 - 2u_i + u_i+1) / h^2 
	const double divisor = grid.spacing(dimension) * grid.spacing(dimension);
	
	double coefficients [stencil_size];

	// Compute Laplacian with no-flux BCs at the interfaces
	auto dom = grid.getDomainIterator();
	while(dom.isNext())
	{
		auto key = dom.get();
		// If point is a boundary point 
		if(grid.template getProp<PHI_SDF>(key) < b_low + grid.spacing(dimension) - std::numeric_limits<double>::epsilon())
		{
			// If left boundary
			if(grid.template getProp<PHI_SDF>(key.move(x, 1)) > grid.template getProp<PHI_SDF>(key))
			{
				coefficients[0] = 0;
				coefficients[1] = -1;
				coefficients[2] = 1;
			}
			// If right boundary
			if(grid.template getProp<PHI_SDF>(key.move(x, -1)) > grid.template getProp<PHI_SDF>(key))
			{
				coefficients[0] = 1;
				coefficients[1] = -1;
				coefficients[2] = 0;
			}
		}
		else // for the inside points use normal central finite difference
		{
			coefficients[0] = 1;
			coefficients[1] = -2;
			coefficients[2] = 1;
		}
		
		finite_differences<FIELD, LAP_U, stencil_size>(grid, key, stencil, coefficients, divisor, dimension);

		++dom;
	}
}


template<size_t FIELD_IN, size_t FIELD_OUT, size_t IS_SOURCE, size_t LAP_U, typename GRID_TYPE, typename FIELD_TYPE>
void source_diffusion_sink(const GRID_TYPE & grid, 
						   const FIELD_TYPE k_source,
						   const FIELD_TYPE k_sink,
						   const FIELD_TYPE diffusion_coefficient, 
						   const FIELD_TYPE time_step)
{
	// Loop over grid and run reaction-diffusion using the concentration laplacian computed above
	auto dom = grid.getDomainIterator();
	while(dom.isNext())
	{
		{
			auto key = dom.get();
			
			// First order Euler time-stepper to update concentration, source-diffusion-degradation
			grid.template get<FIELD_OUT>(key) = 
				grid.template get<FIELD_IN>(key)
				+ time_step * ( 
					k_source * grid.template get<IS_SOURCE>(key)
					+ diffusion_coefficient * grid.template get<LAP_U>(key)
					- k_sink * grid.template get<FIELD_IN>(key));

			
			++dom;
		}
	}
}



int main(int argc, char* argv[])
{
	// Initialize library.
	openfpm_init(&argc, &argv);
	auto & v_cl = create_vcluster();

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Parameters for the reaction-diffusion process (Source-Diffusion-Sink mechanism)
	const double diffusion_coefficient = 2.0; // diffusion constant um2/s
	const double k_source = 0.0; // uM/s
	const double k_sink = 0.0; // 1/s
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Set current working directory, define output paths and create folders where output will be saved
	std::string cwd                     = get_cwd();
	const std::string path_output       = cwd + "/output_1D_SDD/";
	create_directory_if_not_exist(path_output);
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Create grid of size N
	size_t N = 9;
	const size_t sz[dims] = {N};
	const double Lx_low = -1.0;
	const double Lx_up  = 1.0;
	Box<dims, double> box({Lx_low}, {Lx_up});
	Ghost<dims, long int> ghost(0);
	typedef grid_dist_id<dims, double, props> grid_type;
	grid_type g_dist(sz, box, ghost);
	g_dist.setPropNames({"PHI_SDF", "U1_N", "U1_NPLUS1", "LAP_U", "IS_SOURCE"});
	
	
	init_grid_and_ghost<U1_N>(g_dist, 0); // Initialize grid and ghost layer with 0
	init_grid_and_ghost<U1_NPLUS1>(g_dist, 0); // Initialize grid and ghost layer with 0
	init_grid_and_ghost<PHI_SDF>(g_dist, -1); // Initialize grid and ghost layer with -1
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Initialize level-set function with analytic signed distance function at each grid point
	// Initialize whether point is source (30% of the most left grid points)
	auto dom = g_dist.getDomainIterator();
	while(dom.isNext()) // Loop over all grid points
	{
		auto key = dom.get(); // index of current grid node
		
		// Point<grid_type::dims, typename grid_type::stype> coords = g_dist.getPos(key); // get coordinates of grid point
		
		g_dist.template getProp<PHI_SDF>(key) = 1.0 - g_dist.getPos(key).get(x) * g_dist.getPos(key).get(x);

		if(g_dist.getPos(key).get(x) <= Lx_low + 0.3 * (Lx_up - Lx_low) + std::numeric_limits<double>::epsilon())
		{
			g_dist.template getProp<IS_SOURCE>(key) = 1;

			// FOR DEBUGGING -> CHECK IF TOTAL MASS CONSERVED
			g_dist.template getProp<U1_N>(key) = 0.1;
		}
		
		++dom;
	}
	
	g_dist.write(path_output + "grid_initial", FORMAT_BINARY); // Save initial grid
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Get the diffusion timestep that fulfills the stability condition
	const double dx = g_dist.spacing(x), dy = g_dist.spacing(y); // if you want to know the grid spacing
	const double dt = diffusion_time_step(g_dist, diffusion_coefficient);
	std::cout << "dx = " << dx << ", dt = " << dt << std::endl;
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Diffusion using a forward-time central-space scheme
	double t = 0;
	int iter = 0; // initial iteration
	int max_iter = 10; // max iteration
	int interval_write = (int)(max_iter / 10); // set how many frames should be saved as vtk
	
	double b_low = 0; // considered as embryo boundary
	
	while(iter < max_iter)
	{
		std::cout << "-----------------------------iteration " << iter << "----------------------------------" << std::endl;

		// Compute laplacian from UN or UNPLUS1 in even or odd iteration, respectively
		if(iter % 2 == 0)
		{
			// Compute laplacian from UN in even iteration
			finite_differences_2nd_derivative_no_flux_BCs<U1_N, LAP_U, PHI_SDF>(g_dist, x, b_low);
			// Loop over grid and run reaction-diffusion using the concentration laplacian computed above
			source_diffusion_sink<U1_N, U1_NPLUS1, IS_SOURCE, LAP_U>(g_dist, k_source, k_sink, diffusion_coefficient, dt);
		}
		else
		{	
			// Compute laplacian from UNPLUS1 in odd iteration
			finite_differences_2nd_derivative_no_flux_BCs<U1_NPLUS1, LAP_U, PHI_SDF>(g_dist, x, b_low);
			// Loop over grid and run reaction-diffusion using the concentration laplacian computed above
			source_diffusion_sink<U1_NPLUS1, U1_N, IS_SOURCE, LAP_U>(g_dist, k_source, k_sink, diffusion_coefficient, dt);
		}
		
		
		// Write grid to vtk
		if (iter % interval_write == 0)
		{
			g_dist.write_frame(path_output + "/grid_diffuse_withNoFlux", iter, FORMAT_BINARY);
			std::cout << "Diffusion time :" << t << std::endl;
			
			// Monitor total concentration
			monitor_total_concentration<U1_N>(g_dist, t, iter, path_output, "total_conc.csv");

		}
		
		// Update U1_N
		// copy_gridTogrid<U1_NPLUS1, U1_N>(g_dist, g_dist);
		
		iter += 1;
		t += dt;
	}
	
	
	
	
	// g_dist.save(path_output + "/grid_diffuse_withNoFlux" + std::to_string(iter) + ".hdf5"); // Save grid as hdf5 file which can
	// be reloaded for evaluation
	
	
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	openfpm_finalize();
	return 0;
}
