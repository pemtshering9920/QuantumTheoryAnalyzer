import os
import json
import numpy as np
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Dict, Any, List, Optional, Tuple

# Create SQLAlchemy engine using DATABASE_URL environment variable
database_url = os.environ.get('DATABASE_URL')
engine = create_engine(database_url)

# Create base class for SQLAlchemy models
Base = declarative_base()

# Define the SimulationResult model
class SimulationResult(Base):
    __tablename__ = 'simulation_results'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    simulation_type = Column(String(50), nullable=False)
    key_size = Column(Integer, nullable=False)
    error_rate = Column(Float, nullable=False)
    recovery_time = Column(Float, nullable=False)
    initial_fidelity = Column(Float, nullable=False)
    corrupted_fidelity = Column(Float, nullable=False)
    recovered_fidelity = Column(Float, nullable=False)
    fractal_dimension = Column(Float)
    error_type = Column(String(50))
    description = Column(Text)
    parameters = Column(JSONB)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model instance to a dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'simulation_type': self.simulation_type,
            'key_size': self.key_size,
            'error_rate': self.error_rate,
            'recovery_time': self.recovery_time,
            'initial_fidelity': self.initial_fidelity,
            'corrupted_fidelity': self.corrupted_fidelity,
            'recovered_fidelity': self.recovered_fidelity,
            'fractal_dimension': self.fractal_dimension,
            'error_type': self.error_type,
            'description': self.description,
            'parameters': self.parameters
        }

# Create all tables
Base.metadata.create_all(engine)

# Create a session factory
SessionFactory = sessionmaker(bind=engine)

def save_simulation_result(
    simulation_type: str,
    key_size: int,
    error_rate: float,
    recovery_time: float,
    initial_fidelity: float,
    corrupted_fidelity: float,
    recovered_fidelity: float,
    fractal_dimension: Optional[float] = None,
    error_type: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> int:
    """
    Save a simulation result to the database.

    Args:
        simulation_type: Type of simulation (e.g., 'error_correction', 'encryption')
        key_size: Size of the key in bits
        error_rate: Error rate used in the simulation
        recovery_time: Recovery time used in the simulation
        initial_fidelity: Initial fidelity value
        corrupted_fidelity: Fidelity after error introduction
        recovered_fidelity: Fidelity after recovery
        fractal_dimension: Fractal dimension used (if applicable)
        error_type: Type of error introduced (if applicable)
        description: Text description of the simulation
        parameters: Additional parameters as JSON

    Returns:
        The ID of the saved simulation result
    """
    # Create a session
    session = SessionFactory()

    try:
        # Convert numpy values to Python native types
        def convert_np(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
            return obj

        # Convert all numpy values to Python native types
        error_rate = convert_np(error_rate)
        initial_fidelity = convert_np(initial_fidelity)
        corrupted_fidelity = convert_np(corrupted_fidelity)
        recovered_fidelity = convert_np(recovered_fidelity)
        fractal_dimension = convert_np(fractal_dimension) if fractal_dimension is not None else None

        # Convert parameters recursively
        converted_params = {k: convert_np(v) for k, v in parameters.items()} if parameters else None

        # Create a new SimulationResult
        result = SimulationResult(
            simulation_type=simulation_type,
            key_size=key_size,
            error_rate=error_rate,
            recovery_time=recovery_time,
            initial_fidelity=initial_fidelity,
            corrupted_fidelity=corrupted_fidelity,
            recovered_fidelity=recovered_fidelity,
            fractal_dimension=fractal_dimension,
            error_type=error_type,
            description=description,
            parameters=converted_params
        )

        # Add and commit
        session.add(result)
        session.commit()

        # Get the ID
        result_id = result.id

        return result_id

    finally:
        # Always close the session
        session.close()

def get_simulation_results(
    simulation_type: Optional[str] = None,
    limit: int = 10,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Get simulation results from the database.

    Args:
        simulation_type: Filter by simulation type (optional)
        limit: Maximum number of results to return
        offset: Number of results to skip

    Returns:
        List of simulation results as dictionaries
    """
    # Create a session
    session = SessionFactory()

    try:
        # Build query
        query = session.query(SimulationResult)

        # Apply filters
        if simulation_type:
            query = query.filter(SimulationResult.simulation_type == simulation_type)

        # Apply ordering, limit, and offset
        query = query.order_by(SimulationResult.timestamp.desc())
        query = query.limit(limit).offset(offset)

        # Execute query and convert results to dictionaries
        results = [result.to_dict() for result in query.all()]

        return results

    finally:
        # Always close the session
        session.close()

def get_simulation_result_by_id(result_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a simulation result by ID.

    Args:
        result_id: The ID of the simulation result

    Returns:
        The simulation result as a dictionary, or None if not found
    """
    # Create a session
    session = SessionFactory()

    try:
        # Query for the result
        result = session.query(SimulationResult).filter(SimulationResult.id == result_id).first()

        # Return as dictionary if found, None otherwise
        return result.to_dict() if result else None

    finally:
        # Always close the session
        session.close()

def get_simulation_stats() -> Dict[str, Any]:
    """
    Get statistics about the simulation results.

    Returns:
        A dictionary containing statistics
    """
    # Create a session
    session = SessionFactory()

    try:
        # Count total results
        total_count = session.query(func.count(SimulationResult.id)).scalar()

        # Count by simulation type
        type_counts = session.query(
            SimulationResult.simulation_type,
            func.count(SimulationResult.id)
        ).group_by(SimulationResult.simulation_type).all()

        # Get average fidelities
        avg_initial = session.query(func.avg(SimulationResult.initial_fidelity)).scalar() or 0.0
        avg_corrupted = session.query(func.avg(SimulationResult.corrupted_fidelity)).scalar() or 0.0
        avg_recovered = session.query(func.avg(SimulationResult.recovered_fidelity)).scalar() or 0.0

        # Return statistics
        return {
            'total_count': total_count,
            'by_type': dict(type_counts),
            'avg_fidelities': {
                'initial': float(avg_initial),
                'corrupted': float(avg_corrupted),
                'recovered': float(avg_recovered)
            }
        }

    finally:
        # Always close the session
        session.close()

def delete_simulation_result(result_id: int) -> bool:
    """
    Delete a simulation result by ID.

    Args:
        result_id: The ID of the simulation result to delete

    Returns:
        True if successful, False otherwise
    """
    # Create a session
    session = SessionFactory()

    try:
        # Query for the result
        result = session.query(SimulationResult).filter(SimulationResult.id == result_id).first()

        # If not found, return False
        if not result:
            return False

        # Delete the result
        session.delete(result)
        session.commit()

        return True

    finally:
        # Always close the session
        session.close()