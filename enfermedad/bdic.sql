SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";

CREATE TABLE `usuario` (
  `idusuario` int(11) NOT NULL AUTO_INCREMENT,
  `usuario` varchar(50) NOT NULL,
  `clave` varchar(100) NOT NULL,
  `rol` char(1) NOT NULL,
  `estado` char(1) NOT NULL,
  PRIMARY KEY (`idusuario`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `medico` (
  `idmedico` int(11) NOT NULL AUTO_INCREMENT,
  `apellidos` varchar(50) NOT NULL,
  `nombres` varchar(50) NOT NULL,
  `colegiado` varchar(20) NOT NULL,
  `idusuario` int(11) NOT NULL,
  PRIMARY KEY (`idmedico`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `paciente` (
  `idpaciente` int(11) NOT NULL AUTO_INCREMENT,
  `idmedico` int(11) NOT NULL,
  `apellidos` varchar(50) NOT NULL,
  `nombres` varchar(50) NOT NULL,
  `dni` char(8) NOT NULL,
  `fechahora` datetime NOT NULL,
  `idusuario` int(11) NOT NULL,
  PRIMARY KEY (`idpaciente`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `atencion` (
  `idatencion` int(11) NOT NULL AUTO_INCREMENT,
  `idpaciente` int(11) NOT NULL,
  `fechahora` datetime NOT NULL,
  `resultado` varchar(200) NOT NULL,
  PRIMARY KEY (`idatencion`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `sintoma` (
  `idsintoma` int(11) NOT NULL AUTO_INCREMENT,
  `sintoma` varchar(200) NOT NULL,
  `estado` char(1) NOT NULL,
  PRIMARY KEY (`idsintoma`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `atencion_sintoma` (
  `idatencionsintoma` int(11) NOT NULL AUTO_INCREMENT,
  `idatencion` int(11) NOT NULL,
  `idsintoma` int(11) NOT NULL,
  `respuesta` char(1) NOT NULL,
  PRIMARY KEY (`idatencionsintoma`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `enfermedad` (
  `idenfermedad` int(11) NOT NULL AUTO_INCREMENT,
  `nombre` varchar(255) NOT NULL,
  PRIMARY KEY (`idenfermedad`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `enfermedad_sintoma` (
  `idenfermedad` int(11) NOT NULL,
  `idsintoma` int(11) NOT NULL,
  PRIMARY KEY (`idenfermedad`, `idsintoma`),
  KEY `fk_enfermedad_sintoma_sintoma` (`idsintoma`),
  CONSTRAINT `fk_enfermedad_sintoma_enfermedad`
    FOREIGN KEY (`idenfermedad`) REFERENCES `enfermedad` (`idenfermedad`)
    ON DELETE CASCADE,
  CONSTRAINT `fk_enfermedad_sintoma_sintoma`
    FOREIGN KEY (`idsintoma`) REFERENCES `sintoma` (`idsintoma`)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

INSERT INTO `usuario` (`idusuario`, `usuario`, `clave`, `rol`, `estado`) VALUES
(1, 'jperez', '202cb962ac59075b964b07152d234b70', 'm', 'a'),
(2, 'cdiaz', '202cb962ac59075b964b07152d234b70', 'p', 'a');

INSERT INTO `medico` (`idmedico`, `apellidos`, `nombres`, `colegiado`, `idusuario`) VALUES
(1, 'perez', 'juan', '123456', 1);

INSERT INTO `paciente` (`idpaciente`, `idmedico`, `apellidos`, `nombres`, `dni`, `fechahora`, `idusuario`) VALUES
(1, 1, 'diaz', 'carlos', '12345678', '2021-07-18 00:25:10', 2);

INSERT INTO `sintoma` (`idsintoma`, `sintoma`, `estado`) VALUES
(1, 'Fiebre alta de inicio súbito', 'a'),
(2, 'Dolor retroocular (dolor intenso detrás de los globos oculares)', 'a'),
(3, 'Dolor articular severo e incapacitante (dificultad para mover las articulaciones)', 'a'),
(4, 'Conjuntivitis no purulenta (ojos enrojecidos sin secreción ni lagañas)', 'a'),
(5, 'Erupción cutánea o sarpullido (rash maculopapular)', 'a'),
(6, 'Dificultad respiratoria severa y repentina (sensación de ahogo)', 'a'),
(7, 'Dolor muscular agudo, focalizado principalmente en las pantorrillas', 'a'),
(8, 'Ictericia (coloración amarillenta en la piel y la parte blanca de los ojos)', 'a'),
(9, 'Sangrado espontáneo (en mucosas, encías, nariz o bajo la piel)', 'a'),
(10, 'Vómitos persistentes y dolor abdominal intenso', 'a'),
(11, 'Aparición de petequias (pequeños puntos rojos en la piel) o prueba del torniquete positiva', 'a'),
(12, 'Disminución drástica en la cantidad de orina (oliguria)', 'a'),
(13, 'Debilidad muscular progresiva u hormigueo que asciende por las piernas', 'a'),
(14, 'Fotofobia (sensibilidad extrema a la luz)', 'a'),
(15, 'Tos seca persistente acompañada de mareos profundos', 'a');

INSERT INTO `enfermedad` (`idenfermedad`, `nombre`) VALUES
(1, 'DENGUE GRAVE'),
(2, 'SINDROME PULMONAR POR HANTAVIRUS'),
(3, 'LEPTOSPIROSIS (ENFERMEDAD DE WEIL)'),
(4, 'COMPLICACION NEUROLOGICA POR ZIKA (GUILLAIN-BARRE)'),
(5, 'ZIKA'),
(6, 'CHIKUNGUNYA'),
(7, 'DENGUE CLASICO');

INSERT INTO `enfermedad_sintoma` (`idenfermedad`, `idsintoma`) VALUES
(1, 1), (1, 9), (1, 10), (1, 11),
(2, 1), (2, 6), (2, 15),
(3, 1), (3, 7), (3, 8), (3, 12),
(4, 1), (4, 4), (4, 13),
(5, 1), (5, 4), (5, 5),
(6, 1), (6, 3), (6, 5),
(7, 1), (7, 2), (7, 5), (7, 14);

ALTER TABLE `usuario` AUTO_INCREMENT = 3;
ALTER TABLE `medico` AUTO_INCREMENT = 2;
ALTER TABLE `paciente` AUTO_INCREMENT = 2;
ALTER TABLE `atencion` AUTO_INCREMENT = 1;
ALTER TABLE `atencion_sintoma` AUTO_INCREMENT = 1;
ALTER TABLE `sintoma` AUTO_INCREMENT = 16;
ALTER TABLE `enfermedad` AUTO_INCREMENT = 8;
