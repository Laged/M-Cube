#include <SFML/Audio/SoundBuffer.hpp>
#include <fftw3.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

template <class T>
void write(ofstream& fout, T& data)
{
	fout.write((char*)&data, sizeof(T));
}

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		cout << "anna biisu" << endl;
		return -1;
	}
	
	string biisu = argv[1];
	
	sf::SoundBuffer buffer;
	if (!buffer.loadFromFile(biisu.c_str()))
	{
		cout << "biisu ei lataa" << endl;
		return -1;
	}
	
	const sf::Int16* samples = buffer.getSamples();
	int num_samples = buffer.getSampleCount();
	size_t sample_rate = buffer.getSampleRate();
	
	
	if (sample_rate != 44100)
	{
		cout << "sample rate ei oo 44100 noob" << endl;
		return -1;
	}
	
	int samples_per_window = 220; //44100 / 200;
	
	cout << "num samples: " << num_samples << endl;
	
	fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * samples_per_window);
	fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * samples_per_window);
	
	fftw_plan p = fftw_plan_dft_1d(samples_per_window, in, out, FFTW_FORWARD, FFTW_MEASURE);
	
	std::ofstream fout(("ftw/"+biisu+".ftw").c_str(), std::ios::out|std::ios::binary);
	
	write(fout, num_samples);
	write(fout, samples_per_window);
	
	for (int i = 0; i+samples_per_window-1 < num_samples; i += samples_per_window)
	{
		cout << i << endl;
		
		for (int j = 0; j < samples_per_window; ++j)
		{
			in[j][0] = samples[i+j] / 32768.0;
			in[j][1] = 0;
		}
		
		fftw_execute(p);
		
		for (int j = 0; j < samples_per_window; ++j)
		{
			double a = out[j][0];
			double b = out[j][1];
			
			double amp = sqrt(a*a + b*b);
			
			write(fout, amp);
		}
	}
	
	cout << "doned" << endl;
	
	fftw_destroy_plan(p);
	fftw_free(in);
	fftw_free(out);
	
	return 0;
}
