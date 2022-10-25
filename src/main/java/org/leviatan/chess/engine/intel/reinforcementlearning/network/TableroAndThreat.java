package org.leviatan.chess.engine.intel.reinforcementlearning.network;

import org.leviatan.chess.board.Tablero;

/**
 * @author Alejandro
 *
 */
public class TableroAndThreat {

    private final Tablero tablero;
    private double threatBlancas;
    private double threatNegras;

    public TableroAndThreat(final Tablero tablero, final double threatBlancas, final double threatNegras) {
        this.tablero = tablero;
        this.threatBlancas = threatBlancas;
        this.threatNegras = threatNegras;
    }

    public Tablero getTablero() {
        return this.tablero;
    }

    public double getThreatBlancas() {
        return this.threatBlancas;
    }

    public double getThreatNegras() {
        return this.threatNegras;
    }

    public void setThreatBlancasIfSuperior(final double threatBlancas) {

        if (threatBlancas > this.threatBlancas) {
            this.threatBlancas = threatBlancas;
        }
    }

    public void setThreatNegraIfSuperior(final double threatNegras) {

        if (threatNegras > this.threatNegras) {
            this.threatNegras = threatNegras;
        }
    }

}
