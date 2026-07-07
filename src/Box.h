// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#ifndef _BOX
#define _BOX


#include "include_libs.h"
#include "constants.h"

class Box {
    protected:
        int Dim;                            // System dimensionality
        std::string input_command;          // Command to create this box
        std::string boxStyle;               // "ps" or "fts" simulation box
        int nTotalBoxes;                    // Number of simulation boxes

    public:
        thrust::host_vector<int> Nx;        // Grid dimensions
        thrust::device_vector<int> d_Nx;    
        int* _d_Nx;

        thrust::host_vector<double> dx;     // grid spacing in each direction
        thrust::device_vector<double> d_dx;
        double* _d_dx;
        
        float *L, *d_L, *Lh, *d_Lh;         // [Dim] Box dimensions, half-box dimensions

        std::string boxType;

        double V;                           // Box volume
        double Vfree;                       // Box volume not occupied by interfaces, particles
        int M;                              // Total number of grid points
        double gvol;                        // Grid volume
        int M_Grid, M_Block;                // GPU Configuration parameters
        int logFreq;                        // Frequency to print energies
        int densityFieldFreq;               // Frequency to write configs
        int totSteps;                       // Total elapsed iterations
        int maxSteps;                       // Max number of steps to run
        int threads;                        // number of threads per GPU block
        long int simTime;                   // Total simulation time
        long int ftTimer;                   // Time spent in FFT routine
        long int ioTimer;                   // Time in I/O routines
        int blockSize;                      // GPU block size

        cufftHandle fftplan, fftplanSingle; // FFT Plans
        void cufftWrapperDouble(thrust::device_vector<thrust::complex<double>>,
            thrust::device_vector<thrust::complex<double>>&, const int);
        void convolveTComplexDouble(thrust::device_vector<thrust::complex<double>>,
            thrust::device_vector<thrust::complex<double>>&, thrust::device_vector<thrust::complex<double>>);

        void cufftWrapperSingle(cuComplex*, cuComplex*, const int);

        std::complex<double> sumCpxDoubleDeviceArray(cuDoubleComplex*, int, int);


        Box();
        Box(std::istringstream&);
        virtual ~Box();
        void setDimension(int);
        
        // For a given id \in [0,M), defines Fourier vector k
        // Returns |k|^2
        template<typename T>
        T get_kD(int id, T* k) {
            // id between 0 and M-1 (i value in loop), k = kx,ky,kz, Dim is dimensionality
            T kmag = 0.0f;
            int n[3];

            this->unstack2(id, n);

            for (int i = 0; i < Dim; i++) {
                if (float(n[i]) < float(Nx[i]) / 2.)
                    k[i] = PI2 * float(n[i]) / L[i];

                else
                    k[i] = PI2 * float(n[i] - Nx[i]) / L[i];

                kmag += k[i] * k[i];
            }
            return kmag;
        }

        // For given id \in [0,M), defines position vector for that grid point
        template<typename T>
        void get_r(int id, T* r) {
            int n[3];

            this->unstack2(id, n);

            for (int i = 0; i < Dim; i++) {
                r[i] = T(n[i]) * dx[i];
            }
        }

        // For given id \in [0,M), defines position vector for that grid point
        void get_rf(int id, float* r) { get_r(id, r); }

        // Minimum-image separation dr = ri - rj; returns |dr|^2
        template<typename T>
        T pbc_dr2(T* dr, const T* ri, const T* rj) {
            T mdr2 = 0.0;
            for ( int j=0 ; j<Dim ; j++ ) {
                dr[j] = ri[j] - rj[j];

                if ( dr[j] > Lh[j] ) dr[j] -= L[j];
                else if ( dr[j] < -Lh[j] ) dr[j] += L[j];

                mdr2 += dr[j] * dr[j];
            }

            return mdr2;
        }

        void make_bias_field(double*, const double, const std::string, const int, const int);
        
        void unstack2(int, int*);
        int returnDimension(void);
        std::string returnBoxStyle(void);
        void initCuRand(void);
        std::string printCommand();
        virtual void readInput(std::ifstream&) = 0;
        virtual void writeFields() = 0;
        virtual void writeTime() = 0;
        virtual int converged(int) = 0;

        void initBinaryDataFile(std::string);

        // Writes a frame of data to the binary file
        template<typename T>
        void writeBinaryData(std::string name, T* dat) {
            std::ofstream otp(name, std::ios::app|std::ios::binary);
            if ( !otp.is_open() ) {
                die("Failed to open output binary file!");
            }

            otp.write(reinterpret_cast<char*>(dat), M*sizeof(T));

            otp.close();
        }

        // Writes a frame of tensor data to the binary file
        template<typename T>
        void writeBinaryTensorData(std::string name, T* dat) {
            std::ofstream otp(name, std::ios::app|std::ios::binary);
            if ( !otp.is_open() ) {
                die("Failed to open output binary file!");
            }

            otp.write(reinterpret_cast<char*>(dat), Dim*Dim*M*sizeof(T));

            otp.close();
        }

        int known_phase(std::string, std::string&);

        virtual void NVT(int) = 0;

        virtual void writeData(int) = 0;

        virtual void findSpinodal(std::istringstream&) = 0;

        virtual void doTimeStep(int) = 0;

        virtual void modifyBox(std::istringstream&) = 0;

        void readDatFile(std::string, thrust::host_vector<thrust::complex<double>>&);
        // void readDatFile(std::string, float*);

        curandState* d_states;
        int RAND_SEED;

};

#endif // BOX