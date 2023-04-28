from sqlalchemy import create_engine, Column, Integer, Float, String, Enum, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import numpy as np
from typing import Union


class Base(DeclarativeBase):
    pass


class Coordinate(Base):
    __tablename__ = "coordinates"

    id = Column(Integer, primary_key=True)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    z = Column(Float, nullable=False)
    array = Column(np.ndarray, nullable=False)

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.array = np.array([self.x, self.y, self.z])

    def __repr__(self):
        return f"Coordinate({self.x}, {self.y}, {self.z})"

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def distance(self, other):
        return np.linalg.norm(self.array - other.array)

    def angle(self, other1, other2):
        v1 = self.array - other1.array
        v2 = self.array - other2.array
        return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def dihedral(self, other1, other2, other3):
        v1 = self.array - other1.array
        v2 = self.array - other2.array
        v3 = self.array - other3.array
        v12 = np.cross(v1, v2)
        v23 = np.cross(v2, v3)
        return np.arctan2(np.dot(v12, v23), np.dot(np.cross(v12, v23), v2))


class Chain(Base):
    __tablename__ = "chains"

    id = Column(Integer, primary_key=True)
    chain_type = Column(
        Enum("protein", "nucleotide", "other", name="chain_type_enum"), nullable=False
    )
    structure_id = Column(Integer, ForeignKey("structures.id"))

    structure = relationship("Structure", back_populates="chains")
    subunits = relationship("ChainSubunit", back_populates="chain")


class ChainSubunit(Base):
    __abstract__ = True
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    chain_id = Column(Integer, ForeignKey("chains.id"))

    chain = relationship("Chain", back_populates="subunits")


class Residue(ChainSubunit):
    __tablename__ = "residues"

    ca_coord_id = Column(Integer, ForeignKey("coordinates.id"))
    sidechain_com_id = Column(Integer, ForeignKey("coordinates.id"))

    ca_coord = relationship("Coordinate", foreign_keys=[ca_coord_id])
    sidechain_com = relationship("Coordinate", foreign_keys=[sidechain_com_id])


class Nucleotide(ChainSubunit):
    __tablename__ = "nucleotides"

    phosphate_com_id = Column(Integer, ForeignKey("coordinates.id"))
    sugar_com_id = Column(Integer, ForeignKey("coordinates.id"))
    base_com_id = Column(Integer, ForeignKey("coordinates.id"))

    phosphate_com = relationship("Coordinate", foreign_keys=[phosphate_com_id])
    sugar_com = relationship("Coordinate", foreign_keys=[sugar_com_id])
    base_com = relationship("Coordinate", foreign_keys=[base_com_id])


class Chain(Base):
    __tablename__ = "chains"

    id = Column(Integer, primary_key=True)
    chain_type = Column(
        Enum("protein", "nucleotide", "other", name="chain_type_enum"), nullable=False
    )
    structure_id = Column(Integer, ForeignKey("structures.id"))

    structure = relationship("Structure", back_populates="chains")
    subunits = relationship("ChainSubunit", back_populates="chain")


class Structure(Base):
    __tablename__ = "structures"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

    chains = relationship("Chain", back_populates="structure")

    def add_chain_from_dataframe(
        self, chain_type: str, df: pd.DataFrame, session: Session
    ):
        chain = Chain(chain_type=chain_type, structure=self)
        session.add(chain)

        if chain_type == "protein":
            for index, row in df.iterrows():
                ca_coord = Coordinate(*row["ca_coords"])
                sidechain_com = Coordinate(*row["sidechain_coms"])
                residue = Residue(
                    name=row["residue"],
                    ca_coord=ca_coord,
                    sidechain_com=sidechain_com,
                    chain=chain,
                )
                session.add(residue)

        elif chain_type == "nucleotide":
            for index, row in df.iterrows():
                sugar_com = Coordinate(*row["sugar_com"])
                base_com = Coordinate(*row["base_com"])
                phosphate_com = Coordinate(*row["phosphate_com"])
                nucleotide = Nucleotide(
                    name=row["residue"],
                    sugar_com=sugar_com,
                    base_com=base_com,
                    phosphate_com=phosphate_com,
                    chain=chain,
                )
                session.add(nucleotide)

        else:
            raise ValueError(f"Chain type {chain_type} not supported")

        session.commit()
