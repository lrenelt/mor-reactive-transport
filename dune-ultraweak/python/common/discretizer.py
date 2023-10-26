import ipyultraweak as uw

from pymor.operators.constructions import LincombOperator, VectorOperator
from pymor.parameters.functionals import ExpressionParameterFunctional

from pymor_classes import DuneOperator, DuneUltraweakTransportModel

def makeLincombOperatorFromDune(coeffFunctionals, duneOperatorList):
    operators = [DuneOperator(op) for op in duneOperatorList]

    return LincombOperator(operators, coeffFunctionals, name='DuneLincombOperator')


def discretize(config, solverConfig, order=1):
    # generate solver object holding eg Grid or the GridFunctionSpaces
    if order==1:
        solverObj = uw.TransportSolverOrder1(config)
    elif order==2:
        solverObj = uw.TransportSolverOrder2(config)
    else:
        raise NotImplementedError("Only order 1 and 2 implemented")

    # TODO: pass these to the solver
    coeffFunctionals = [1.0,
                        ExpressionParameterFunctional('Acw[0]', parameters={'Acw': 1}),
                        ExpressionParameterFunctional('Bcc[0]', parameters={'Bcc': 1}),
                        ExpressionParameterFunctional('Acw[0]', parameters={'Acw': 1}),
                        ExpressionParameterFunctional('Acw[0]**2', parameters={'Acw': 1}),
                        ExpressionParameterFunctional('Acw[0]*Bcc[0]',
                                                      parameters={'Acw': 1, 'Bcc': 1}),
                        ExpressionParameterFunctional('Bcc[0]', parameters={'Bcc': 1}),
                        ExpressionParameterFunctional('Bcc[0]*Acw[0]',
                                                      parameters={'Acw': 1, 'Bcc': 1}),
                        ExpressionParameterFunctional('Bcc[0]**2', parameters={'Bcc': 1}),
                        ]
    operator = makeLincombOperatorFromDune(coeffFunctionals, solverObj.getOperators())

    coeffFunctionalsRhs = [1.0,
                        ExpressionParameterFunctional('Cgin[0]', parameters={'Cgin': 1})]
    rhs_operators = [VectorOperator(operator.range.make_array([rhsVec])) for rhsVec in solverObj.getRhsVectors()]
    rhs = LincombOperator(rhs_operators, coeffFunctionalsRhs, name="DuneLincombOperatorRhs")

    # build the model
    return DuneUltraweakTransportModel(operator, rhs, solverObj, solverConfig)
