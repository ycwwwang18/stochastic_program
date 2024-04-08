from gurobipy import *
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(a=100)  # 设置随机生成器的种子，保证可重复性
np.random.seed(100)


class Subproblem:
    def __init__(self, revisit_status, lagrangian_multipliers_1, lagrangian_multipliers_2, rho, i_1, i_2, k):
        """
        revisit_status: the revisit status of first visit patients with revisit, offline or online, dimension: i_1
        lagrangian_multipliers: mu, dimensions: j, t, i_1 or i_2
        rho: rho_1, rho_2 are the penalty parameters in Lagrangian relaxation
        consensus_para: x^hat, dimensions: j, t, i_1 or i_2
        """
        self.revisit_status = revisit_status
        self.lagrangian_multipliers_1 = lagrangian_multipliers_1
        self.lagrangian_multipliers_2 = lagrangian_multipliers_2
        self.rho = rho
        self.I_1 = i_1
        self.I_2 = i_2
        self.k = k
        self.J = 4  # num of physician
        self.MWTT_FV_with_RV = 3  # the appointment maximum wait time target of FV with RV patients
        self.MWTT_FV_without_RV = 5  # the appointment maximum wait time target of FV without RV patients
        self.total_capacity = 480  # the total daily capacity of each physician
        self.online_capacities = np.array([  # online capacities of four physicians in the planning horizon
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [60, 60, 60, 60],
            [60, 60, 60, 60],
            [60, 60, 60, 60],
            [60, 60, 60, 60],
            [60, 60, 60, 60],
            [60, 60, 60, 60],
            [60, 60, 60, 60],
            [60, 60, 60, 60],
            [60, 60, 60, 60],
            [60, 60, 60, 60],
            [60, 60, 60, 60],
            [60, 60, 60, 60],
            [60, 60, 60, 60]
        ])
        self.online_capacities = self.online_capacities.T
        self.offline_capacities = np.array([  # offline capacities of four physicians in the planning horizon
            [240, 240, 240, 240],
            [240, 240, 240, 240],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420]
        ])
        self.offline_capacities = self.offline_capacities.T
        self.min_rv_interval_off = 2  # minimum revisit interval of offline
        self.max_rv_interval_off = 3  # maximum revisit interval of offline
        self.min_rv_interval_on = 4  # minimum revisit interval of online
        self.max_rv_interval_on = 5  # maximum revisit interval of online
        self.K = 5  # arriving horizon
        self.L = 12  # planning horizon
        self.T = self.K + self.L
        self.service_duration_fv_with_rv = 60  # service duration for a FV with RV, minutes
        self.service_duration_on_rv = 30  # service duration for online revisit
        self.service_duration_off_rv = 40  # service duration for offline revisit
        self.service_duration_fv_without_rv = 40  # service duration for a FV without RV
        self.continuity_violation_off = 1.5  # continuity of care violation penalties for offline
        self.continuity_violation_on = 1  # continuity of care violation penalties for online
        self.over_idle_cost_off = 1.5  # offline overtime or idle costs
        self.over_idle_cost_on = 1  # online overtime or idle costs
        self.setup_cost_off = 1.5  # fixed offline service setup cost
        self.setup_cost_on = 1  # fixed online service setup cost
        self.online_prob = 0.6  # the probability of online revisit

        # sets
        self.arriving_horizon = range(self.K)  # set of periods within an arriving horizon
        self.planning_horizon = range(self.L)  # set of periods in the L-day planning window
        self.FV_with_RV = range(self.I_1)  # set of FV with RV patients
        self.FV_without_RV = range(self.I_2)  # set of FV without RV patients
        self.physicians = range(self.J)  # set of physicians

        # infinite positive numbers
        self.M = 99999999

    def run_model(self):
        # 建模
        try:
            """创建模型"""
            m = Model("ss")

            """创建变量"""
            assign_fv_with_rv = m.addVars(self.I_1, self.J, self.T, vtype=GRB.BINARY, name='x1')
            assign_fv_without_rv = m.addVars(self.I_2, self.J, self.T, vtype=GRB.BINARY, name='x2')
            assign_off = m.addVars(self.I_1, self.J, self.T, vtype=GRB.BINARY, name='y')
            assign_on = m.addVars(self.I_1, self.J, self.T, vtype=GRB.BINARY, name='z')
            overtime_off = m.addVars(self.J, self.T, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name='o1')
            idle_off = m.addVars(self.J, self.T, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name='a1')
            overtime_on = m.addVars(self.J, self.T, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name='o2')
            idle_on = m.addVars(self.J, self.T, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name='a2')
            consensus_para_fv_with_rv = m.addVars(self.I_1, self.J, self.T, vtype=GRB.BINARY, name='x1_hat')
            consensus_para_fv_without_rv = m.addVars(self.I_2, self.J, self.T, vtype=GRB.BINARY, name='x2_hat')

            # initial the consensus parameters
            for i_1 in self.FV_with_RV:
                for j in self.physicians:
                    for t in range(self.T):
                        consensus_para_fv_with_rv[i_1, j, t].start = 0

            for i_2 in self.FV_without_RV:
                for j in self.physicians:
                    for t in range(self.T):
                        consensus_para_fv_without_rv[i_2, j, t].start = 0

            m.update()

            """目标函数"""
            # total overtime or idle time cost
            over_idle_cost_off = quicksum(
                self.over_idle_cost_off * (overtime_off[j, t] + idle_off[j, t])
                for j in self.physicians for t in range(self.T))
            over_idle_cost_on = quicksum(
                self.over_idle_cost_on * (overtime_on[j, t] + idle_on[j, t])
                for j in self.physicians for t in range(self.T))
            set_up_cost = quicksum(
                self.setup_cost_off * self.offline_capacities[j, t] + self.setup_cost_on * self.online_capacities[j, t]
                for j in self.physicians for t in range(self.T)
            )
            continuity_cost_off = self.continuity_violation_off * quicksum(
                self.revisit_status[i_1] *
                (quicksum(assign_off[i_1, j, t] for t in range(self.T)) - quicksum(
                    assign_fv_with_rv[i_1, j, t] for t in range(self.T)))
                for i_1 in self.FV_with_RV for j in self.physicians)
            continuity_cost_on = self.continuity_violation_on * quicksum(
                (1 - self.revisit_status[i_1]) *
                (quicksum(assign_on[i_1, j, t] for t in range(self.T)) -
                 quicksum(assign_fv_with_rv[i_1, j, t] for t in range(self.T)))
                for i_1 in self.FV_with_RV for j in self.physicians)
            la_11 = quicksum(self.lagrangian_multipliers_1[i_1, j, t] * (
                    assign_fv_with_rv[i_1, j, t] - consensus_para_fv_with_rv[i_1, j, t]) for i_1 in self.FV_with_RV
                             for j in self.physicians for t in range(self.k + 1, self.k + self.L + 1))
            la_12 = self.rho[0] / 2 * quicksum(
                assign_fv_with_rv[i_1, j, t] - 2 * assign_fv_with_rv[i_1, j, t] * consensus_para_fv_with_rv[i_1, j, t] +
                consensus_para_fv_with_rv[i_1, j, t] for i_1 in self.FV_with_RV for j in self.physicians for t in
                range(self.k + 1, self.k + self.L + 1))
            la_21 = quicksum(self.lagrangian_multipliers_2[i_2, j, t] * (
                    assign_fv_without_rv[i_2, j, t] - consensus_para_fv_without_rv[i_2, j, t]) for i_2 in
                             self.FV_without_RV for j in self.physicians for t in
                             range(self.k + 1, self.k + self.L + 1))
            la_22 = self.rho[1] / 2 * quicksum(
                assign_fv_without_rv[i_2, j, t] - 2 * assign_fv_without_rv[i_2, j, t] * consensus_para_fv_without_rv[
                    i_2, j, t] + consensus_para_fv_without_rv[i_2, j, t] for i_2 in self.FV_without_RV for j in
                self.physicians for t in range(self.k + 1, self.k + self.L + 1))

            obj_value = over_idle_cost_off + over_idle_cost_on + set_up_cost + continuity_cost_off + continuity_cost_on + la_11 + la_12 + la_21 + la_22
            m.setObjective(obj_value, GRB.MINIMIZE)

            """创建约束"""
            # FV with RV patients first visit 1 time during planning horizon
            m.addConstrs(quicksum(assign_fv_with_rv[i_1, j, t] for j in self.physicians for t in
                                  range(self.k + 1, self.k + self.L + 1)) == 1 for i_1 in self.FV_with_RV)
            # FV without RV patients first visit 1 time during planning horizon
            m.addConstrs(quicksum(assign_fv_without_rv[i_2, j, t] for j in self.physicians for t in
                                  range(self.k + 1, self.k + self.L + 1)) == 1 for i_2 in self.FV_without_RV)
            # FV with RV patients revisit 1 time offline if revisit status is offline
            m.addConstrs(quicksum(
                assign_off[i_1, j, t] for j in self.physicians for t in range(self.k + 1, self.k + self.L + 1)) ==
                         self.revisit_status[i_1] for i_1 in self.FV_with_RV)
            # FV with RV patients revisit 1 time online if revisit status is online
            m.addConstrs(quicksum(
                assign_on[i_1, j, t] for j in self.physicians for t in range(self.k + 1, self.k + self.L + 1)) ==
                         1 - self.revisit_status[i_1] for i_1 in self.FV_with_RV)
            # offline capacity restrictions
            m.addConstrs(quicksum(assign_fv_with_rv[i_1, j, t] for i_1 in self.FV_with_RV) +
                         quicksum(assign_fv_without_rv[i_2, j, t] for i_2 in self.FV_without_RV) +
                         quicksum(assign_off[i_1, j, t] for i_1 in self.FV_with_RV)
                         <= self.offline_capacities[j, t]
                         for j in self.physicians for t in range(self.k+1, self.k+self.L+1))
            # online capacity restrictions
            m.addConstrs(quicksum(assign_on[i_1, j, t] for i_1 in self.FV_with_RV) <= self.online_capacities[j, t]
                         for j in self.physicians for t in range(self.k+1, self.k+self.L+1))
            # offline revisit time window restrictions
            m.addConstrs(
                quicksum(t * assign_off[i_1, j, t] for j in self.physicians for t in range(self.k+1, self.k+self.L+1)) >=
                quicksum(t * assign_fv_with_rv[i_1, j, t] for j in self.physicians for t in range(self.k+1, self.k+self.L+1)) +
                self.min_rv_interval_off - self.M*(1 - self.revisit_status[i_1]) for i_1 in self.FV_with_RV
            )
            m.addConstrs(
                quicksum(t * assign_off[i_1, j, t] for j in self.physicians for t in range(self.k+1, self.k+self.L+1)) <=
                quicksum(t * assign_fv_with_rv[i_1, j, t] for j in self.physicians for t in range(self.k + 1, self.k + self.L + 1)) +
                self.max_rv_interval_off + self.M*(1 - self.revisit_status[i_1]) for i_1 in self.FV_with_RV
            )
            # online revisit time window restrictions
            m.addConstrs(
                quicksum(t * assign_on[i_1, j, t] for j in self.physicians for t in
                         range(self.k + 1, self.k + self.L + 1)) >=
                quicksum(t * assign_fv_with_rv[i_1, j, t] for j in self.physicians for t in
                         range(self.k + 1, self.k + self.L + 1)) +
                self.min_rv_interval_on - self.M * self.revisit_status[i_1] for i_1 in self.FV_with_RV
            )
            m.addConstrs(
                quicksum(t * assign_on[i_1, j, t] for j in self.physicians for t in
                         range(self.k + 1, self.k + self.L + 1)) <=
                quicksum(t * assign_fv_with_rv[i_1, j, t] for j in self.physicians for t in
                         range(self.k + 1, self.k + self.L + 1)) +
                self.max_rv_interval_on + self.M * self.revisit_status[i_1] for i_1 in self.FV_with_RV
            )
            # first visit with RV time window restrictions
            m.addConstrs(quicksum(t * assign_fv_with_rv[i_1, j, t] for j in self.physicians for t in
                         range(self.k + 1, self.k + self.L + 1)) <= self.MWTT_FV_with_RV for i_1 in self.FV_with_RV)
            # first visit without RV time window restrictions
            m.addConstrs(quicksum(t * assign_fv_without_rv[i_2, j, t] for j in self.physicians for t in
                         range(self.k + 1, self.k + self.L + 1)) <= self.MWTT_FV_without_RV for i_2 in self.FV_without_RV)
            # calculate overtime for offline
            m.addConstrs(
                overtime_off[j, t] >=
                quicksum(self.service_duration_fv_with_rv*assign_fv_with_rv[i_1, j, t] for i_1 in self.FV_with_RV) +
                quicksum(self.service_duration_off_rv*assign_off[i_1, j, t] for i_1 in self.FV_with_RV) +
                quicksum(self.service_duration_fv_without_rv*assign_fv_without_rv[i_2, j, t] for i_2 in self.FV_without_RV)
                - self.offline_capacities[j, t] for j in self.physicians for t in range(self.k+1, self.k+self.L+1)
            )
            # calculate idle time for offline
            m.addConstrs(
                idle_off[j, t] >= self.offline_capacities[j, t] -
                quicksum(self.service_duration_fv_with_rv*assign_fv_with_rv[i_1, j, t] for i_1 in self.FV_with_RV) -
                quicksum(self.service_duration_off_rv * assign_off[i_1, j, t] for i_1 in self.FV_with_RV) -
                quicksum(
                    self.service_duration_fv_without_rv * assign_fv_without_rv[i_2, j, t] for i_2 in self.FV_without_RV)
                for j in self.physicians for t in range(self.k + 1, self.k + self.L + 1)
            )
            # calculate overtime for online
            m.addConstrs(
                overtime_on[j, t] >=
                quicksum(self.service_duration_on_rv*assign_on[i_1, j, t] for i_1 in self.FV_with_RV) -
                self.online_capacities[j, t] for j in self.physicians for t in range(self.k + 1, self.k + self.L + 1)
            )
            # calculate idle time for online
            m.addConstrs(
                idle_on[j, t] >=
                self.online_capacities[j, t] -
                quicksum(self.service_duration_on_rv*assign_on[i_1, j, t] for i_1 in self.FV_with_RV)
                for j in self.physicians for t in range(self.k + 1, self.k + self.L + 1)
            )

            """模型求解"""
            m.Params.LogToConsole = True  # 显示求解过程
            m.Params.TimeLimit = 100  # 限制求解时间为1000s
            m.optimize()

            """输出变量值"""
            if m.Status == GRB.OPTIMAL:
                opt_x1 = np.empty(shape=(self.I_1, self.J, self.T))
                for i_1 in self.FV_with_RV:
                    for j in self.physicians:
                        for t in range(self.T):
                            opt_x1[i_1, j, t] = assign_fv_with_rv[(i_1, j, t)].X
                opt_x2 = np.empty(shape=(self.I_2, self.J, self.T))
                for i_2 in self.FV_without_RV:
                    for j in self.physicians:
                        for t in range(self.T):
                            opt_x2[i_2, j, t] = assign_fv_without_rv[(i_2, j, t)].X
                opt_y_off = np.empty(shape=(self.I_1, self.J, self.T))
                for i_1 in self.FV_with_RV:
                    for j in self.physicians:
                        for t in range(self.T):
                            opt_y_off[i_1, j, t] = assign_off[(i_1, j, t)].X
                opt_z_on = np.empty(shape=(self.I_1, self.J, self.T))
                for i_1 in self.FV_with_RV:
                    for j in self.physicians:
                        for t in range(self.T):
                            opt_z_on[i_1, j, t] = assign_on[(i_1, j, t)].X
                opt_overtime_off = np.empty(shape=(self.J, self.T))
                for j in self.physicians:
                    for t in range(self.T):
                        opt_overtime_off[j, t] = overtime_off[(j, t)].X
                opt_idle_off = np.empty(shape=(self.J, self.T))
                for j in self.physicians:
                    for t in range(self.T):
                        opt_idle_off[j, t] = idle_off[(j, t)].X
                opt_overtime_on = np.empty(shape=(self.J, self.T))
                for j in self.physicians:
                    for t in range(self.T):
                        opt_overtime_on[j, t] = overtime_on[(j, t)].X
                opt_idle_on = np.empty(shape=(self.J, self.T))
                for j in self.physicians:
                    for t in range(self.T):
                        opt_idle_on[j, t] = idle_on[(j, t)].X
                obj_value = m.ObjVal
                return opt_x1, opt_x2, opt_y_off, opt_z_on, (opt_overtime_off, opt_idle_off, opt_overtime_on, opt_idle_on, obj_value)
            else:
                return None

        except GurobiError as e:
            print('Error code ' + str(e.errno) + ':' + str(e))

        except AttributeError:
            print('Encountered an attribute error')

    def result_capacity_analysis(self, opt_x1, opt_x2, opt_y_off, opt_z_on):
        """输出每个医生的容量分配"""
        fv_with_rv = opt_x1.sum(axis=0)
        fv_without_rv = opt_x2.sum(axis=0)
        rv_off = opt_y_off.sum(axis=0)
        rv_on = opt_z_on.sum(axis=0)

        p_1_capacity_assign = np.empty(shape=(self.L, 4))
        p_2_capacity_assign = np.empty(shape=(self.L, 4))
        p_3_capacity_assign = np.empty(shape=(self.L, 4))
        p_4_capacity_assign = np.empty(shape=(self.L, 4))

        # physician 1
        for k in range(self.k + 1, self.k + self.L + 1):
            p_1_capacity_assign[k - self.k - 1, 0] = fv_with_rv[0, k]
            p_1_capacity_assign[k - self.k - 1, 1] = fv_without_rv[0, k]
            p_1_capacity_assign[k - self.k - 1, 2] = rv_off[0, k]
            p_1_capacity_assign[k - self.k - 1, 3] = rv_on[0, k]

        # physician 2
        for k in range(self.k + 1, self.k + self.L + 1):
            p_2_capacity_assign[k - self.k - 1, 0] = fv_with_rv[1, k]
            p_2_capacity_assign[k - self.k - 1, 1] = fv_without_rv[1, k]
            p_2_capacity_assign[k - self.k - 1, 2] = rv_off[1, k]
            p_2_capacity_assign[k - self.k - 1, 3] = rv_on[1, k]

        # physician 3
        for k in range(self.k + 1, self.k + self.L + 1):
            p_3_capacity_assign[k - self.k - 1, 0] = fv_with_rv[2, k]
            p_3_capacity_assign[k - self.k - 1, 1] = fv_without_rv[2, k]
            p_3_capacity_assign[k - self.k - 1, 2] = rv_off[2, k]
            p_3_capacity_assign[k - self.k - 1, 3] = rv_on[2, k]

        # physician 4
        for k in range(self.k + 1, self.k + self.L + 1):
            p_4_capacity_assign[k - self.k - 1, 0] = fv_with_rv[3, k]
            p_4_capacity_assign[k - self.k - 1, 1] = fv_without_rv[3, k]
            p_4_capacity_assign[k - self.k - 1, 2] = rv_off[3, k]
            p_4_capacity_assign[k - self.k - 1, 3] = rv_on[3, k]

        # 作图
        # physician 1
        x = np.arange(p_1_capacity_assign.shape[0])  # 横坐标的索引
        bottom = np.zeros_like(p_1_capacity_assign[:,0])  # 初始化底部高度为0
        label = ['FV with RV', 'FV without RV', 'Offline Revisit', 'Online Revisit']
        for i in range(p_1_capacity_assign.shape[1]):
            plt.bar(x, p_1_capacity_assign[:, i], width=0.45, bottom=bottom, label=label[i])
            bottom += p_1_capacity_assign[:, i]  # 更新底部高度
        plt.legend()
        plt.xlabel('Planning horizon(day)')
        plt.ylabel('Capacity(min)')
        plt.title('Physician One')
        plt.show()

        # physician 2
        x = np.arange(p_2_capacity_assign.shape[0])  # 横坐标的索引
        bottom = np.zeros_like(p_2_capacity_assign[:, 0])  # 初始化底部高度为0
        for i in range(p_2_capacity_assign.shape[1]):
            plt.bar(x, p_2_capacity_assign[:, i], width=0.45, bottom=bottom, label=label[i])
            bottom += p_2_capacity_assign[:, i]  # 更新底部高度
        plt.legend()
        plt.xlabel('Planning horizon(day)')
        plt.ylabel('Capacity(min)')
        plt.title('Physician Two')
        plt.show()

        # physician 3
        x = np.arange(p_3_capacity_assign.shape[0])  # 横坐标的索引
        bottom = np.zeros_like(p_3_capacity_assign[:, 0])  # 初始化底部高度为0
        for i in range(p_3_capacity_assign.shape[1]):
            plt.bar(x, p_3_capacity_assign[:, i], width=0.45, bottom=bottom, label=label[i])
            bottom += p_3_capacity_assign[:, i]  # 更新底部高度
        plt.legend()
        plt.xlabel('Planning horizon(day)')
        plt.ylabel('Capacity(min)')
        plt.title('Physician Three')
        plt.show()

        # physician 4
        x = np.arange(p_4_capacity_assign.shape[0])  # 横坐标的索引
        bottom = np.zeros_like(p_4_capacity_assign[:, 0])  # 初始化底部高度为0
        for i in range(p_4_capacity_assign.shape[1]):
            plt.bar(x, p_4_capacity_assign[:, i], width=0.45, bottom=bottom, label=label[i])
            bottom += p_4_capacity_assign[:, i]  # 更新底部高度
        plt.legend()
        plt.xlabel('Planning horizon(day)')
        plt.ylabel('Capacity(min)')
        plt.title('Physician Four')
        plt.show()


if __name__ ==  "__main__":
    i_1 = np.random.poisson(lam=28.5)  # average daily numbers of patient FV with RV is 28.5
    i_2 = np.random.poisson(lam=18.1)  # average daily numbers of patient FV without RV is 18.1
    k = 1
    revisit_status = np.random.binomial(1, 0.6, i_1)
    lagrangian_multipliers_1 = np.zeros(shape=(i_1, 4, 17))
    lagrangian_multipliers_2 = np.zeros(shape=(i_2, 4, 17))
    rho = [0, 0]

    sub_problem = Subproblem(revisit_status, lagrangian_multipliers_1, lagrangian_multipliers_2, rho, i_1, i_2, k)
    opt_x1, opt_x2, opt_y_off, opt_z_on, _ = sub_problem.run_model()
    sub_problem.result_capacity_analysis(opt_x1, opt_x2, opt_y_off, opt_z_on)