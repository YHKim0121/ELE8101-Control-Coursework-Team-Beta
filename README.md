**# ELE8101-Control-Coursework-Team-Beta**

Supervisor: Dr. Pantelis Sopasakis
Team: Yonghyeon Kim (40489830), Ming Foong Lau (40348411), Yousif Mustafa Elnaim Hamad (40500333)


**What this is?**
A 7-state Extended Kalman Filter for vehicle localisation on a piecewise racetrack, built using Python and CasADi. The estimator tracks longitudinal progress, lateral offset, and vertical suspension dynamics simultaneously, while compensating for a systematic sensor bias in real time.

We followed the four-step incremental approach from the coursework spec — each member independently prototyped their own version at each step, then the team reviewed and merged the best ideas into a single final implementation.


**Repo layout**
Step1_Brainstorm_Code/ — individual Step 1 prototypes
Step1_Final_Code/ — Step1_CW_Final_Code.ipynb (1D linear Kalman filter)

Step2_Brainstrom_Code/ — individual Step 2 prototypes
Step2_Final_Code/ — Step2_CW_Final_Code.ipynb (EKF on circular track, arc length parameterisation)

Step3_Brainstrom_Code/ — individual Step 3 prototypes
Step3_Final_Code/ — Step3_Final_Code.ipynb (4-state EKF with Ornstein-Uhlenbeck lateral dynamics)

Step4_Brainstorm_Code/ — individual Step 4 prototypes
Step4_Final_Code/ — Step4_Final_Code.ipynb (7-state EKF, full piecewise track, MSD + bias augmentation)

Technical Report/ — final PDF submission


**Setup**
Python 3.10+ required.

git clone https://github.com/YHKim0121/ELE8101-Control-Coursework-Team-Beta.git
cd ELE8101-Control-Coursework-Team-Beta
pip install -r requirements.txt


**How to Run**
Final code (Steps 1–4) — all final implementations are Jupyter notebooks. Launch Jupyter and open the relevant notebook:

jupyter notebook

Then open the notebook in the browser:

Step1_Final_Code/Step1_CW_Final_Code.ipynb
Step2_Final_Code/Step2_CW_Final_Code.ipynb
Step3_Final_Code/Step3_Final_Code.ipynb
Step4_Final_Code/Step4_Final_Code.ipynb

Run all cells top to bottom (Kernel → Restart & Run All). For the main results, Step 4 prints the full RMSE summary and generates the trajectory and estimation plots inline.

Brainstorm code — individual member prototypes are in the Brainstorm_Code folders. File formats vary by member (.ipynb, .py, and plain text). Notebooks open the same way as above; .py files can be run directly from terminal:

python filename.py

When you run Step4_Final_Code.ipynb, you should see something like:

Execution time : ~0.4–0.7 ms
RMSE lateral : ~0.12 m
Final bias error: ~0.003 m


**Notes**
Random seed is fixed (np.random.seed(42) in Step 4) so results are reproducible

CasADi handles all Jacobian computation, so no manual Jacobian derivation is needed in the final code

The final beacon positions are B1(-60, 65), B2(-60, -65), B3(130, 65), and B4(130, -65)

Bias augmentation is applied to B1 only (1.5 m offset) as a proof of concept


**Collaboration**
Branching strategy was agreed at the first meeting — each member worked in their own branch and nothing was merged to main without a pull request review. Meeting logs and task tracking are in Appendix B–E of the report, and we also kept a shared Notion workspace during the project.
