import pandas as pd
from scipy.stats import pearsonr, linregress
import matplotlib.pyplot as plt
import numpy as np


data = {
    'case': [1]*7*4+[3]*7*2+[4]*7*3+[5]*7+[6]*7+[9]*7+[10]*7*3+[11]*7+[12]*7+[13]*7+[15]*7*2+[16]*7*2+[17]*7*2,
    'branch':['Proximal LAD']*7+['Mid_LAD']*7+['Septal']*7+['1st Diag']*7+['Proximal LAD']*7+['Mid LCx']*7+['Proximal LAD']*7+['Distal LAD']*7+['Distal RCA']*7+['Proximal RCA']*7+['Distal RCA']*7+['Proximal LAD']*7+['Left Main']*7+['Mid Proximal LAD']*7+['Mid Distal LAD']*7+['Left Main']*7+['Mid distal LAD']*7+['Distal to Mid LAD']*7+['Proximal LAD']*7+['Distal LAD']*7+['Mid Prox RCA']*7+['Mid distal RCA']*7+['Distal LAD']*7+['PDA']*7,
    'constant':[2.0,2.2,2.4,2.6,2.8,3.0,2.66]*7*13*24,
    'Mean Pressure (mmHg)':[110.4871236, 109.9712473, 110.6419056, 108.8115419, 109.4583369, 108.9035784, 110.7629286, 110.1292331, 109.6311754, 110.270098, 108.4606661, 109.1391181, 108.4282628, 110.1044519, 109.9117463, 109.4273766, 110.0626372, 108.2678695, 108.9684422, 108.1795446, 109.7328584, 109.9878276, 109.4966747, 110.1245981, 108.3218422, 109.0161716, 108.2618954, 109.8849768, 78.44293409, 78.50997852, 78.68600639, 79.74387557, 80.28364324, 76.57897398, 80.77889279, 78.66599892, 78.69644135, 78.86276817, 79.83801238, 80.42195428, 76.68278055, 81.04364596, 80.1852363, 80.02609317, 80.87118551, 79.89641569, 80.74444811, 80.95702161, 85.83261323, 76.36078612, 76.05537421, 77.04944472, 76.17278911, 76.9440546, 77.29076982, 83.5783531, 78.94334612, 78.72153196, 79.5745143, 78.59531745, 79.3863167, 79.60045345, 84.55580948, 101.740879, 101.6131487, 104.5034044, 85.51824432, 102.7166848, 102.7220186, 106.3180725, 146.6999722, 89.78125859, 145.0679623, 91.8132021, 88.03795387, 91.45926753, 90.77373754, 112.4465012, 113.7510623, 112.6725391, 113.3555977, 112.3328784, 111.9230142, 116.7087986, 110.4843721, 113.0210066, 114.206303, 112.3552877, 111.6242038, 114.9795053, 112.4752769, 110.2871545, 112.8162721, 114.0114447, 112.1621374, 111.4476665, 114.8016258, 112.1811203, 109.9909366, 112.5228587, 113.7183246, 111.895256, 111.2123619, 114.5488699, 111.6863842, 95.7210377, 97.47192754, 97.20269773, 97.99761888, 96.67597355, 99.07254445, 99.04714275, 73.14529641, 74.33087954, 75.25449864, 74.11059732, 74.02134248, 75.50016859, 72.47651971, 89.38655012, 90.13798759, 98.82625815, 90.264607, 97.37514751, 91.09986375, 92.48269297, 80.73005918, 81.57541492, 83.24854668, 81.40951106, 80.56869043, 81.31168772, 82.09636664, 78.41065106, 79.16067614, 80.8937441, 79.09998825, 78.09731933, 79.06678289, 79.96213239, 75.03376036, 74.88288166, 75.57408559, 73.91147744, 74.02406344, 73.23948703, 75.2026908, 74.94454089, 74.78999756, 75.47202686, 73.80654171, 73.84529763, 73.1196579, 75.11716159, 92.76369952, 151.5518183, 146.8777153, 92.71824467, 151.6850071, 92.36638674, 91.36798433, 94.66337352, 153.5163381, 148.4701006, 94.21384312, 152.9026444, 93.73607998, 92.91464376],
    'Invasive Pressure (mmHg)':[110.3]*7*4+[82.25]*7*2+[80.51]*7*3+[102.9]*7+[87.24]*7+[110]*7+[106.85]*7*3+[96.03]*7+[71.04]*7+[91.64]*7+[76.99]*7*2+[72.04]*7*2+[90]*7*2,
}

data_2 = {
    'case': [1]*4+[3]*2+[4]*3+[5]+[6]+[9]+[10]*3+[11]+[12]+[13]+[15]*2+[16]*2,
    'branch': ['Proximal LAD']+['Mid_LAD']+['Septal']+['1st Diag']+['Proximal LAD']+['Mid LCx']+['Proximal LAD']+['Distal LAD']+['Distal RCA']+['Proximal RCA']+['Distal RCA']+['Left Main']+['Mid Proximal LAD']+['Mid Distal LAD']+['Left Main']+['Mid distal LAD']+['Distal to Mid LAD']+['Proximal LAD']+['Distal LAD']+['Mid Prox RCA']+['Mid distal RCA'],
    'constant': [2.6]*11*24,
    'Mean Pressure (mmHg)': [108.811541879093, 108.4606661, 108.2678695, 108.3218422, 79.74387557, 79.83801238, 79.89641569, 76.17278911, 78.59531745, 85.51824432, 91.8132021, 112.3552877, 112.1621374, 111.895256, 97.99761888, 74.11059732, 90.264607, 81.40951106, 79.09998825, 73.91147744, 73.80654171],
    'Invasive Pressure (mmHg)':[110.3]*4+[82.25]*2+[80.51]*3+[102.9]+[87.24]+[106.85]*3+[96.03]+[71.04]+[91.64]+[76.99]*2+[72.04]*2,
}

pressure_CFD = np.array(data_2['Mean Pressure (mmHg)'])
pressure_Invasive = np.array(data_2['Invasive Pressure (mmHg)'])

assert len(pressure_CFD) == len(pressure_Invasive)

correlation_coefficient, p_value = pearsonr(pressure_CFD, pressure_Invasive)

print(f"Pearson correlation coefficient: {correlation_coefficient:.3f}")
print(f"P-value: {p_value}")

slope, intercept, r_value, p_value, std_err = linregress(pressure_CFD, pressure_Invasive)
print(f"Regression slope: {slope:.2f}, intercept: {intercept:.2f}, r-squared: {r_value**2:.2f}")



plt.scatter(pressure_CFD, pressure_Invasive)
plt.plot(pressure_CFD, slope * pressure_CFD + intercept, color='red')
plt.plot(pressure_CFD, pressure_CFD, linestyle='--', color='gray')
plt.xlabel('Pressure CFD (mmHg)')
plt.ylabel('Invasive Pressure (mmHg)')
plt.title('Correlation between Simulation Results and Measured Data')

#plt.text(0.5, 5.5, f'P-value: {p_value}', fontsize=10, ha='left', va='center', color='green')
#plt.text(0.5, 5.2, f'R-squared: {r_value**2:.4f}', fontsize=10, ha='left', va='center', color='purple')

plt.legend([f'P-value: {p_value}',f'R-squared: {r_value**2:.4f}'])
plt.grid(True)
plt.tight_layout()
plt.show()
